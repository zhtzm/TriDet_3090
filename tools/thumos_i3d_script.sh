#!/bin/bash

echo "start training"
CUDA_VISIBLE_DEVICES=$1 python train.py ./configs/thumos_i3d.yaml --output pretrained
echo "start testing..."
CUDA_VISIBLE_DEVICES=$1 python eval.py ./configs/thumos_i3d.yaml ckpt/thumos_i3d_pretrained/epoch_039.pth.tar

<< results
|tIoU = 0.10: mAP = 87.10 (%)
|tIoU = 0.20: mAP = 86.24 (%)
|tIoU = 0.30: mAP = 83.63 (%)
|tIoU = 0.40: mAP = 80.33 (%)
|tIoU = 0.50: mAP = 73.04 (%)
|tIoU = 0.60: mAP = 62.16 (%)
|tIoU = 0.70: mAP = 46.58 (%)
Avearge mAP: 74.15 (%) (69.15 (%))