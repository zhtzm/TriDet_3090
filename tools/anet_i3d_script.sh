##!/bin/bash

echo "start training"
CUDA_VISIBLE_DEVICES=$1 python train.py ./configs/anet_i3d.yaml --output pretrained
echo "start testing..."
CUDA_VISIBLE_DEVICES=$1 python eval.py ./configs/anet_i3d.yaml ckpt/anet_i3d_pretrained/epoch_014.pth.tar
