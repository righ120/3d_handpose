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

모델은 총 세 개의 하위 모델들로 이루어져 있다. 
(1) HandSegNet에서는 이미지에서 양 손은 찾는다.
(2) PoseNet 에서는 손 이미지에서 21개의 keypoints를 찾는다. 각각의 keypoints는 손바닥, 엄지, 손목... 등등을 가리킨다.
(3) PriorPose 에서는 2D - keypoints를 3D 로 매핑하는 역할을 한다.

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


