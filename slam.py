
import sys
import torch
import cv2
import numpy as np
import glob
import os
import time
import yaml
from utils.graphics_utils import focal2fov,getWorld2View2_tensor,getProjectionMatrix2
from utils.slam_utils import get_median_depth
from utils.pose_utils import update_pose
from utils.so3_utils import SE3_exp
from gaussian_renderer import render
from scene import GaussianModel
from scene.camera_class import Camera
from argparse import ArgumentParser
from utils.config_utils import load_config

def get_dataset(path):
    c_paths = sorted(
        glob.glob(f'{path}/color/*.jpg'))

    d_paths = sorted(
        glob.glob(f'{path}/depth/*.png'))

    n = len(c_paths)

    return c_paths, d_paths, n

class SLAM:
    def __init__(self, config):
        """
        SLAM Process

        """

        # Bring the params
        self.config = config
        self.dataset = config['Dataset']
        self.training = config['Training']

        self.opt_params = config['opt_params']
        self.model_params = config['model_params']
        self.pipeline_params = config['pipeline_params']

        self.feature_params = config['feature_params']

        self.device_name = config['model_params']['data_device']
        self.down_scale = config['Dataset']['downscale']
        self.depth_scale = config['Dataset']['Calibration']['depth_scale']
        self.window_num = config['Training']['window_size']
        self.lam = config['opt_params']['lambda_dssim']
    
        self.fx = config['Dataset']['Calibration']['fx'] / self.down_scale
        self.fy = config['Dataset']['Calibration']['fy'] / self.down_scale
        self.cx = config['Dataset']['Calibration']['cx'] / self.down_scale
        self.cy = config['Dataset']['Calibration']['cy'] / self.down_scale
        self.height = config['Dataset']['Calibration']['height'] / self.down_scale
        self.width = config['Dataset']['Calibration']['width'] / self.down_scale

        self.global_cnt = 0

        self.images = None # Color images
        self.depths = None # Depth images
        self.num = None # The number of data frames
        self.gaussians = None # Gaussians
        self.keyframe_list = []
        self.last_idx= -1 # last window index
        self.current_idx = 0 # current window index
        self.prev_R = torch.eye(3)
        self.prev_T = torch.zeros(3)
        self.cur_cam = None
        self.is_keyframe = False
        self.background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")

        #Dataset Load
        dataset_path = config['Dataset']['dataset_path']

        self.images, self.depths, self.num = get_dataset(dataset_path)


    def initialization(self):
        """
        Map initialization
        """
        gt_color = self.cur_cam.original_image
        gt_depth = self.cur_cam.depth
        H,W =gt_depth.shape
        gt_depth_tensor = torch.Tensor(gt_depth).to(device=self.device_name)

        #Initialize the gaussians
        self.gaussians = GaussianModel(self.config['model_params']['sh_degree'])
        self.gaussians.create_gaussians(self.cur_cam,H,W,0,initial=True)
        self.gaussians.training_setup(self.opt_params)

        for init_iter in range(self.training['init_itr_num']):
            self.global_cnt+=1
            render_pkg = render(self.cur_cam,self.gaussians,self.pipeline_params,self.background)
            render_color = render_pkg['color']
            render_depth = render_pkg["depth"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii =  render_pkg["radii"]

            color_mask = (gt_color.sum(dim=0)>0.01).view(*gt_depth_tensor.shape)
            depth_mask = (gt_depth_tensor>0.01).view(*gt_depth_tensor.shape)

            loss_color = torch.abs(gt_color*color_mask-render_color*color_mask)
            loss_depth = torch.abs(gt_depth_tensor*depth_mask-render_depth*depth_mask)
            loss_init= self.lam*loss_color.mean() + (1-self.lam)*loss_depth.mean()
            #print('Loss is',loss_init)
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )

                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if init_iter%self.training['init_gaussian_update']==0:
                    self.gaussians.densify_and_prune(
                        self.opt_params['densify_grad_threshold'],
                        self.training['init_gaussian_th'],
                        self.training['init_gaussian_extent']*self.opt_params['spatial_lr_scale'],
                        None
                    )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        keyframe = []
        keyframe.append(self.cur_cam)
        self.keyframe_list.append(keyframe)


    def frontend(self) -> bool:
        """ Tracking and calculate co-visibility for keyframe selection

        Returns:
            bool: Tracked frame is keyframe: True
                  Tracked frame is not keyframe: False
        """
        v_cam = self.cur_cam

        l = [
                {'params': [v_cam.cam_rot_delta], 'lr': 0.003, "name": "rotation"},
                {'params': [v_cam.cam_trans_delta], 'lr': 0.001, "name": "translation"},
                {'params': [v_cam.exposure_a], 'lr': 0.01, "name": "exposure_a"},
                {'params': [v_cam.exposure_b], 'lr': 0.01, "name": "exposure_b"}
            ]
        optimizer = torch.optim.Adam(l)
        v_cam.compute_grad_mask()
        lam = self.lam

        for track_iter in range(self.training['tracking_itr_num']):
            render_pkg = render(v_cam,self.gaussians,self.pipeline_params,self.background)
            render_color = render_pkg["color"]
            render_depth = render_pkg["depth"]
            render_opacity = render_pkg["opacity"]

            optimizer.zero_grad()

            gt_color = v_cam.original_image
            gt_depth = torch.Tensor(v_cam.depth).to(device=self.device_name)
            _,h,w = gt_color.shape
            mask_shape = (1,h,w)

            rgb_pixel_mask = (gt_color.sum(dim=0) > self.training['rgb_boundary_threshold']).view(*mask_shape)
            rgb_pixel_mask = rgb_pixel_mask * v_cam.grad_mask
            color_exp = (torch.exp(v_cam.exposure_a)) * gt_color + v_cam.exposure_b
            loss_color = render_opacity*torch.abs(color_exp*rgb_pixel_mask-render_color*rgb_pixel_mask)

            depth_mask = (gt_depth>0.01).view(*gt_depth.shape)
            opacity_mask = (render_opacity > 0.95).view(*render_depth.shape)
            depth_mask = depth_mask * opacity_mask

            loss_depth = torch.abs(gt_depth*depth_mask-render_depth*depth_mask)
            loss= lam*loss_color.mean() + (1-lam)*loss_depth.mean()
            loss.backward(retain_graph=True)

            with torch.no_grad():
                optimizer.step()
                is_converge=update_pose(self.cur_cam)

            if is_converge:
                break

        self.prev_R = v_cam.R
        self.prev_T = v_cam.T

        current_window_size = len(self.keyframe_list[self.current_idx])

        if(current_window_size==0):
            last_keyframe= self.keyframe_list[self.last_idx][-1]
        else:
            last_keyframe = self.keyframe_list[self.current_idx][current_window_size-1]
        last_W2C=getWorld2View2_tensor(last_keyframe.R,last_keyframe.T)
        last_C2W=torch.linalg.inv(last_W2C)



        cur_visibility = (render_pkg["n_touched"]>0).long()
        keyframe_render = render(v_cam,self.gaussians,self.pipeline_params,self.background)
        keyframe_visibility = (keyframe_render["n_touched"]>0).long()

        intersection = torch.logical_and(cur_visibility,keyframe_visibility).count_nonzero()
        union = torch.logical_or(cur_visibility,keyframe_visibility).count_nonzero()
        ratio = intersection / union

        cur_W2C = getWorld2View2_tensor(v_cam.R,v_cam.T)
        relative_t = torch.norm((last_C2W@cur_W2C)[0:3,3])
        median_depth = get_median_depth(render_depth, render_opacity)

        dist1 = relative_t > 0.04 * median_depth
        dist2 = relative_t > 0.02 * median_depth

        if((ratio<0.95 and dist2) or dist1):
            return True
        else:
            return False



    def backend(self):
        """
        Map optimization
        """
        current_window = self.keyframe_list[self.current_idx]

        for mapping_iter in range(self.training['mapping_itr_num']):
            self.global_cnt+=1
            densify=False
            keyframe_bundle=[]
            optimize_keyframes=[]
            view_tensor=[]
            visibility_filter=[]
            radii =[]
            n_touch=[]

            for i in range(len(current_window)):
                optimize_keyframes.append(current_window[i])
            if(self.current_idx>0):
                for i in range(self.current_idx):
                    for k in range(self.window_num):
                        keyframe_bundle.append(self.keyframe_list[i][k])
                random_windows=torch.randperm(len(keyframe_bundle))[:2]
                optimize_keyframes.append(keyframe_bundle[random_windows[0]])
                optimize_keyframes.append(keyframe_bundle[random_windows[1]])              
            isotropic_loss=0
            loss_map=0
            p=[]
            
            for id in range(len(optimize_keyframes)):
                p.append({'params': [optimize_keyframes[id].cam_rot_delta], 'lr': 0.003*0.5, "name": "delta_rotation"})
                p.append({'params': [optimize_keyframes[id].cam_trans_delta], 'lr': 0.001*0.5, "name": "delta_translation"})
                p.append({'params': [optimize_keyframes[id].exposure_a], 'lr': 0.01, "name": "exposure_a"})
                p.append({'params': [optimize_keyframes[id].exposure_b], 'lr': 0.01, "name": "exposure_b"})
            pose = torch.optim.Adam(p)

            for cam in enumerate(optimize_keyframes):
                render_pkg = render(cam[1],self.gaussians, self.pipeline_params, self.background)
                render_color = render_pkg["color"]
                render_depth = render_pkg["depth"]

                view_tensor.append(render_pkg["viewspace_points"])
                visibility_filter.append(render_pkg["visibility_filter"])
                radii.append(render_pkg["radii"])
                n_touch.append(render_pkg["n_touched"])

                origin_color = torch.Tensor(cam[1].original_image)
                origin_depth= torch.Tensor(cam[1].depth)
                origin_color=origin_color.cuda()
                origin_depth=origin_depth.cuda()

                color_mask = (origin_color.sum(dim=0)>0.01).view(*origin_depth.shape)
                depth_mask = (origin_depth>0.01).view(*origin_depth.shape)
                color_exp = (torch.exp(cam[1].exposure_a)) * render_color + cam[1].exposure_b

                loss_c =torch.abs(origin_color*color_mask-color_exp*color_mask)
                loss_d =torch.abs(origin_depth*depth_mask-render_depth*depth_mask)
                lam = 0.95
                loss_map += lam*loss_c.mean()+(1-lam)*loss_d.mean()
            scaling = self.gaussians.get_scaling
            isotropic_loss += torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))

            loss_map+=10*isotropic_loss.mean()
            loss_map.backward()

            with torch.no_grad():
                for k in range(len(visibility_filter)):
                    self.gaussians.max_radii2D[visibility_filter[k]] = torch.max(self.gaussians.max_radii2D[visibility_filter[k]],
                                                                            radii[k][visibility_filter[k]],)
                    self.gaussians.add_densification_stats(
                        view_tensor[k], visibility_filter[k]
                    )

                if((self.global_cnt % self.training['gaussian_update_every'])==self.training['gaussian_update_offset']):

                    self.gaussians.densify_and_prune(
                        self.opt_params['densify_grad_threshold'],
                        self.training['gaussian_th'],
                        self.training['gaussian_extent']*self.opt_params['spatial_lr_scale'],
                        self.training['size_threshold']
                    )
                    densify=True

                if(self.global_cnt%self.training['gaussian_reset'])==0 and densify==False:
                    print('Reset!')
                    self.gaussians.reset_opacity_nonvisible(visibility_filter)

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.global_cnt)

                pose.step()
                pose.zero_grad(set_to_none=True)

                for cam in enumerate(optimize_keyframes):
                    with torch.no_grad():
                        if cam[1].uid == 0:
                            continue
                        update_pose(cam[1])
            self.prev_R=self.cur_cam.R
            self.prev_T=self.cur_cam.T

        if(len(self.keyframe_list[self.current_idx])==self.window_num):
            self.keyframe_list.append([])
            self.last_idx+=1
            self.current_idx+=1



    def run(self):
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            W=self.width,
            H=self.height,
        ).transpose(0, 1)

        FovX = focal2fov(self.fx, self.width)
        FovY = focal2fov(self.fy, self.height)

        print('The number of the frame is ',self.num)

        for idx in range(self.num):
            color_img = cv2.imread(self.images[idx])
            depth_img = cv2.imread(self.depths[idx],cv2.IMREAD_UNCHANGED)
            depth_img = depth_img/ self.depth_scale

            H, W = depth_img.shape
            d_H = int(H/self.down_scale)
            d_W = int(W/self.down_scale)

            #Resize & convert the images
            color_img = cv2.resize(color_img, dsize=(d_W,d_H))
            depth_img = cv2.resize(depth_img, dsize=(d_W,d_H), interpolation=cv2.INTER_NEAREST)
            depth_tensor = torch.Tensor(depth_img).to(device=self.device_name)

            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            color_img = torch.Tensor(color_img.transpose(2, 0, 1)).to(device=self.device_name)
            color_img = color_img/255

            g_T =np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

            self.cur_cam = Camera(idx, color_img, depth_img, g_T, projection_matrix, self.fx,self.fy, self.cx, self.cy,
                                  FovX,FovY, d_H,d_W, device = self.device_name)
            self.cur_cam.update_RT(self.prev_R,self.prev_T)

            if(idx==0):
                self.initialization()
                continue

            print('Tracking idx ',idx)
            is_keyframe = self.frontend()

            # If tracked frame is keyframe, create gaussians and map optimization
            if(is_keyframe):
                print('{} is Keyframe!'.format(idx))
                print('New gaussians insertion')

                new_gaussians = GaussianModel(self.config['model_params']['sh_degree'])
                new_gaussians.create_gaussians(self.cur_cam, self.cur_cam.image_height,self.cur_cam.image_width,idx)

                self.gaussians.densification_postfix(
                    new_gaussians._xyz,
                    new_gaussians._features_dc,
                    new_gaussians._features_rest,
                    new_gaussians._opacity,
                    new_gaussians._scaling,
                    new_gaussians._rotation,
                    new_gaussians.gaussian_idx,
                )

                self.keyframe_list[self.current_idx].append(self.cur_cam)
                self.backend()

        model_path = './'
        point_cloud_path = os.path.join(model_path, "output/result/point_cloud/iteration_{}".format(1))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


if __name__ == "__main__":
    parser = ArgumentParser(description="Params")
    parser.add_argument("--config", type=str)
    args = parser.parse_args(sys.argv[1:])

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    slam= SLAM(config)

    slam.run()


