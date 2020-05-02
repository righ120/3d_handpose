# 3d_handpose

Paper: [Learning to Estimate 3D Hand Pose from Single RGB Images](https://arxiv.org/pdf/1705.01389v3.pdf)

# Implementation
## Tested Environments
Environment system (tested):
- Windows 10 
- Pytorch 1.5.0 GPU build with CUDA 10.2 and CUDNN 7.6.5
- Python 3.7

Python packages used by the example provided:
- matplotlib
- numpy
- skimage
- torchvision

## Overall Model Architecture

���� �� �� ���� ���� �𵨵�� �̷���� �ִ�. 
(1) HandSegNet������ �̹������� �� ���� ã�´�.
(2) PoseNet ������ �� �̹������� 21���� keypoints�� ã�´�. ������ keypoints�� �չٴ�, ����, �ո�... ����� ����Ų��.
(3) PriorPose ������ 2D - keypoints�� 3D �� �����ϴ� ������ �Ѵ�.

### HandSegNet
 
<img src="/img/Seg_1.png" width="450px" height="300px" title="Input" alt="RubberDuck"></img><br/>
<img src="/img/Seg_2.png" width="450px" height="300px" title="Input" alt="RubberDuck"></img><br/>
<img src="/img/Seg_3.png" width="450px" height="300px" title="Input" alt="RubberDuck"></img><br/>
### CropImage
### PoseNet
### PriorPose

# To do
* Weight Initialize : 
* Evaluation
	* GestureNet : 


