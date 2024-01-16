##!/bin/bash

echo "start training"
# python train.py ./configs/anet_i3d_aformer.yaml --output aformer200
# python train.py ./configs/anet_i3d.yaml --output pretrained
echo "start testing..."
# python eval.py ./configs/anet_i3d.yaml ckpt/anet_i3d_tridet200/epoch_014.pth.tar
python eval.py ./configs/anet_i3d_aformer.yaml ckpt/anet_i3d_aformer_aformer200/epoch_014.pth.tar

<< results

aformer 1 classes
|tIoU = 0.50: mAP = 54.24 (%)
|tIoU = 0.55: mAP = 50.86 (%)
|tIoU = 0.60: mAP = 47.55 (%)
|tIoU = 0.65: mAP = 44.48 (%)
|tIoU = 0.70: mAP = 40.62 (%)
|tIoU = 0.75: mAP = 36.24 (%)
|tIoU = 0.80: mAP = 31.17 (%)
|tIoU = 0.85: mAP = 24.76 (%)
|tIoU = 0.90: mAP = 17.70 (%)
|tIoU = 0.95: mAP = 8.68 (%)
Avearge mAP: 35.63 (%)

tridet 1 classes
|tIoU = 0.50: mAP = 54.29 (%)
|tIoU = 0.55: mAP = 51.25 (%)
|tIoU = 0.60: mAP = 47.98 (%)
|tIoU = 0.65: mAP = 44.83 (%)
|tIoU = 0.70: mAP = 40.88 (%)
|tIoU = 0.75: mAP = 36.69 (%)
|tIoU = 0.80: mAP = 31.68 (%)
|tIoU = 0.85: mAP = 25.88 (%)
|tIoU = 0.90: mAP = 18.53 (%)
|tIoU = 0.95: mAP = 8.25 (%)
Avearge mAP: 36.03 (%)


tridet200 classes
|tIoU = 0.50: mAP = 46.31 (%)
|tIoU = 0.55: mAP = 43.40 (%)
|tIoU = 0.60: mAP = 40.66 (%)
|tIoU = 0.65: mAP = 37.52 (%)
|tIoU = 0.70: mAP = 33.95 (%)
|tIoU = 0.75: mAP = 29.88 (%)
|tIoU = 0.80: mAP = 25.29 (%)
|tIoU = 0.85: mAP = 19.87 (%)
|tIoU = 0.90: mAP = 13.69 (%)
|tIoU = 0.95: mAP = 5.86 (%)
Avearge mAP: 29.64 (%)

aformer200
|tIoU = 0.50: mAP = 44.96 (%)
|tIoU = 0.55: mAP = 42.36 (%)
|tIoU = 0.60: mAP = 39.63 (%)
|tIoU = 0.65: mAP = 36.83 (%)
|tIoU = 0.70: mAP = 33.62 (%)
|tIoU = 0.75: mAP = 30.50 (%)
|tIoU = 0.80: mAP = 26.08 (%)
|tIoU = 0.85: mAP = 21.38 (%)
|tIoU = 0.90: mAP = 14.78 (%)
|tIoU = 0.95: mAP = 6.29 (%)
Avearge mAP: 29.64 (%)