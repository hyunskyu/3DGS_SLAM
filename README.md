# 3DGS_SLAM

## This is not an original repository. 

This repository is implementation practice of the paper "Gaussian Splatting SLAM[CVPR 2024]"  
  
The original repository go to [https://github.com/muskie82/MonoGS].  



### Not implementated yet
Real-time gui  
Keyframe management  

# Installation
```
git clone https://github.com/hyunskyu/3DGS_SLAM.git
cd 3DGS_SLAM
```

# Requirements
```
conda env create -f environment.yml
conda activate 3dgs_slam
```

# Dataset
Test in only replica office0
```python
bash ./scripts/download_replica.sh
```
# How to run it
```
python slam.py --config configs/replica/office0.yaml
```

# Viewer
```python
./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m output/result
```
