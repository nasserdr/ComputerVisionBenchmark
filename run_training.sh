#!/bin/bash

#archs=(faster_rcnn_R_101_FPN_3x.yaml faster_rcnn_X_101_32x8d_FPN_3x.yaml retinanet_R_101_FPN_3x.yaml)
archs=(
#faster_rcnn_R_50_C4_1x.yaml
3faster_rcnn_R_50_DC5_1x.yaml
faster_rcnn_R_50_FPN_1x.yaml 
retinanet_R_50_FPN_1x.yaml
faster_rcnn_R_50_FPN_3x.yaml
retinanet_R_50_FPN_3x.yaml
faster_rcnn_R_50_DC5_3x.yaml
faster_rcnn_R_50_C4_3x.yaml
retinanet_R_101_FPN_3x.yaml
faster_rcnn_X_101_32x8d_FPN_3x.yaml
faster_rcnn_R_101_C4_3x.yaml
faster_rcnn_R_101_DC5_3x.yaml
faster_rcnn_R_101_FPN_3x.yaml)


IMS_PER_BATCH=(8 16 32 64)
MAX_ITER=(500)

for arch in "${archs[@]}"; do
  for batch_size in "${IMS_PER_BATCH[@]}"; do
    for iter in "${MAX_ITER[@]}"; do
      python train.py --arch "$arch" --ipb "$batch_size" --iter "$iter"
    done
  done
done
