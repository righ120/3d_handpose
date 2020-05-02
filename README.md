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
<img src="img/model_architecture.png" width="900px" height="300px" alt="model_architecture"></img><br/>

**(image) -> HandSegNet -> (hand_mask) -> CropAndResize -> (cropped_hand_image) -> PoseNet -> (hand_keypoints_map) + (hand_side) -> PriorPose -> (hand_keypoints_3d_coords, rotation_matrix)**

���� �� �� ���� ���� �𵨵�� �̷���� �ִ�. <br/>
(1) HandSegNet������ �̹������� �� ���� ã�´�. <br/>
(2) PoseNet ������ �� �̹������� 21���� keypoints�� ã�´�. ������ keypoints�� �չٴ�, ����, �ո�... ����� ����Ų��.<br/>
(3) PriorPose ������ 2D - keypoints�� 3D �� �����ϴ� ������ �Ѵ�.<br/>


### HandSegNet
 
<img src="img/Seg_1.png" width="900px" height="300px" alt="Seg_1"></img><br/>
<img src="img/Seg_2.png" width="900px" height="300px" alt="Seg_2"></img><br/>
<img src="img/Seg_3.png" width="900px" height="300px" alt="Seg_3"></img><br/>

**Input (3x256x256)**: RHD dataset �������� 360x360 �̹����� �Է¹޾� 256x256 ũ��� �����ϰ� �߶󳽴�. <br/> 
**Label (256x256)**: Mask data�� ����� 0, ����� 1, �޼��� 2-17, �������� 18- �� ǥ�� �Ǿ� �ִ�. ���⼭ �� ���� 1 �������� 0���� ó���Ѵ�.<br/>

Loss: �� �ȼ� �� BinaryCrossEntropy Loss�� �����Ͽ� �н���Ų��.<br/>

### CropImage
<img src="img/Crop_1.png" width="300px" height="600px" alt="Crop_1"></img>
<img src="img/Crop_2.png" width="300px" height="600px" alt="Crop_2"></img>
<img src="img/Crop_3.png" width="300px" height="600px" alt="Crop_3"></img><br/>

HandSegNet�� ����� �������� �̹������� �ո��� �߶󳽴�. <br/>
Training �ÿ� �����հ� �޼� �߿��� �� ū ������ �����ϴ� �ո��� �߶󳽴�.

### PoseNet
<img src="img/Scoremap_1.jpg" width="900px" height="600px" alt="Scoremap_1"></img><br/>
<img src="img/Scoremap_2.jpg" width="900px" height="600px" alt="Scoremap_2"></img><br/>
<img src="img/Scoremap_3.jpg" width="900px" height="600px" alt="Scoremap_3"></img><br/>

**Input (3x256x256)**: �߶� ���� �̹����� 256x256���� Scaling �Ѵ�. ���� keypoint ��ǥ���� �̿� �°� �������ش�. (���� ��� �̹����� ������ ��)<br/>
**Label (256x256x21)**: scoremap �̶�� Ī�ϸ�, keypoint ��ǥ�� �߽������Ͽ� ����þ� ���·� 0~1������ ���� ������. 256x256 �� ���� ����ũ�� keypoint �Ѱ��� mapping�ǹǷ� �� 21���� scoremap�� �ִ�. <br/>
**Output (3x32x32x21)**: Convolution Network�� PoseNet ���� �߰� layer 2���� ������ layer�� output���� �Ѵ�.<br/>

**Loss**: �� ���� output�� label���� mse loss�� �����Ͽ� �н���Ų��.<br/>

### PriorPose
**Input (32x32x21)**: PoseNet 3���� output�� AveragePooling �� ���� �Է����� �Ѵ�. ���� �߰� layer�� ����� ������ concat �����ش�.(�޼�:[1,0], ������:[0,1]) <br/>
**Label (21x3, 3x3)**: 21���� keypoint ��ǥ�� Rotation Matrix<br/>
**Output (63, 3x3)**: 2���� convolution network stream�� output�� ���� �����Ѵ�.<br/>

**Loss**: ���� �ΰ��� output�� label ���� mse loss�� �����Ͽ� �н���Ų��. <br/>

# To do
* Weight Initialize : ������ pretrain �� segmentation model�� ����ġ ���� �ʱ�ȭ ��Ų �� �н��Ѵٰ� �Ͽ�����, pytorch�� �н��� ���� ��� �ð��� �����Ͽ���.
* Evaluation : ������ ���� �߰������� �����ؾ� �ϴ� �ڵ�� �ð��� �����Ͽ���.
* �پ��� augmentation ��� Ȥ�� noise �߰�


