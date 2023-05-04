# Download Detectron 2 code:
git clone https://github.com/facebookresearch/Detectron.git

# From the following link, follow the steps to install caffe2:
https://docs.huihoo.com/caffe/caffe2/docs/getting-started.html?platform=windows&configuration=compile

pip install future  numpy  protobuf six hypothesis
pip install flask glog graphviz jupyter matplotlib pydot python-nvd3 pyyaml requests scikit-image scipy setuptools tornado


# Benchmark Logic
We have the possibility to create the following GPU profiles:

•	A100D-1-10C = 1 Copy Engines, 10 GB GPU Memory
•	A100D-2-20C = 2 Copy Engines, 20 GB GPU Memory
•	A100D-3-40C = 3 Copy Engines, 40 GB GPU Memory
•	A100D-4-40C = 4 Copy Engines, 40 GB GPU Memory
•	A100D-7-80C = 7 Copy Engines, 80 GB GPU Memory (= full GPU)

The computing power can go from 1 to 7 copy engines and this should affect the execution time.
The memory can go from 10 to 80 GB and this should affect which model and how many images we can
run through the training.

With the available dectectron2 framework, we have the possibility to run different models:
faster_rcnn_R_101_C4_3x.yaml
faster_rcnn_R_101_DC5_3x.yaml
faster_rcnn_R_101_FPN_3x.yaml 
faster_rcnn_R_50_C4_1x.yaml
faster_rcnn_R_50_C4_3x.yaml 
faster_rcnn_R_50_DC5_1x.yaml 
faster_rcnn_R_50_DC5_3x.yaml 
faster_rcnn_R_50_FPN_1x.yaml 
faster_rcnn_R_50_FPN_3x.yaml 
faster_rcnn_X_101_32x8d_FPN_3x.yaml 
retinanet_R_101_FPN_3x.yaml 
retinanet_R_50_FPN_1x.yaml 
retinanet_R_50_FPN_3x.yaml 
rpn_R_50_C4_1x.yaml 
rpn_R_50_FPN_1x.yaml

Details about these models could be found in the model zoo: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

Models contain different bacjbones. 
- Feature Pyramid Network (FPN) is an architecture for efficiently building high-level semantic feature maps at different scales. FPN is used in Faster R-CNN and other object detection models to improve the detection performance across a wide range of object scales.
- Use a ResNet conv4 backbone with conv5 head. The original baseline in the Faster R-CNN paper.
- DC5 (Dilated-C5): Use a ResNet conv5 backbone with dilations in conv5, and standard conv and FC heads for mask and box prediction, respectively. This is used by the Deformable ConvNet paper.

By elimination, I would keep the FPN option because:
- FPN give a better time/accuracy tradeoff.
- The DC5 is used for segmentation which we do not have in the case study.
- The RPN is not a full network.

faster_rcnn_R_101_FPN_3x.yaml 
faster_rcnn_R_50_FPN_1x.yaml 
faster_rcnn_R_50_FPN_3x.yaml 
faster_rcnn_X_101_32x8d_FPN_3x.yaml 
retinanet_R_101_FPN_3x.yaml 
retinanet_R_50_FPN_1x.yaml 
retinanet_R_50_FPN_3x.yaml 

From the above list, I would choose only the following bigger configurations (probably lower configurations in case the memory is not good for lower GPU profiles):
faster_rcnn_R_101_FPN_3x.yaml 
faster_rcnn_X_101_32x8d_FPN_3x.yaml 
retinanet_R_101_FPN_3x.yaml

# First benchmark executed on 03/05/2023. Results are stored in results_A100D-7-80C