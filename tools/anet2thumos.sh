python crd_eval.py ./configs/anet_i3d_aformer.yaml ckpt/anet_i3d_aformer/epoch_014.pth.tar a2t class_mapping/a2t_class_mapping.json ./configs/thumos_i3d_aformer.yaml

<< dd
python crd_eval.py ./configs/anet_i3d.yaml ckpt/anet_i3d_tridet/epoch_014.pth.tar a2t class_mapping/a2t_class_mapping.json ./configs/thumos_i3d.yaml

<< results
SmD:
|tIoU = 0.50: mAP = 57.95 (%)
|tIoU = 0.55: mAP = 56.73 (%)
|tIoU = 0.60: mAP = 55.92 (%)
|tIoU = 0.65: mAP = 47.06 (%)
|tIoU = 0.70: mAP = 43.74 (%)
|tIoU = 0.75: mAP = 37.98 (%)
|tIoU = 0.80: mAP = 30.36 (%)
|tIoU = 0.85: mAP = 22.22 (%)
|tIoU = 0.90: mAP = 10.08 (%)
|tIoU = 0.95: mAP = 3.47 (%)
Avearge mAP: 36.55 (%)
CrD:
|tIoU = 0.10: mAP = 56.68 (%)
|tIoU = 0.20: mAP = 47.36 (%)
|tIoU = 0.30: mAP = 30.74 (%)
|tIoU = 0.40: mAP = 20.61 (%)
|tIoU = 0.50: mAP = 13.85 (%)
|tIoU = 0.60: mAP = 9.14 (%)
|tIoU = 0.70: mAP = 5.18 (%)
Avearge mAP: 26.22 (%)

