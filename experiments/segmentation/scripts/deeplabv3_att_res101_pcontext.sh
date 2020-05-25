# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model deeplabv3_att --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname deeplabv3_att_res101_pcontext

#test [single-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model deeplabv3_att --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/deeplabv3_att/deeplabv3_att_res101_pcontext/checkpoint.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model deeplabv3_att --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/deeplabv3_att/deeplabv3_att_res101_pcontext/checkpoint.pth.tar --split val --mode testval --ms