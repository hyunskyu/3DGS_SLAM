# 3DGS_SLAM

## This is not a original repository. 

This repository is implementation practice of the paper "Gaussian Splatting SLAM[CVPR 2024]"  
  
The original repository go to [https://github.com/muskie82/MonoGS].  



### Not implementated yet
Real-time gui  
Keyframe management  


# Dataset
Test in only replica office0
```python
cd 3DGS_SLAM
bash ./scripts/download_replica.sh
```
# How to run it
```
python slam.py --config configs/replica/office0.yaml
```
