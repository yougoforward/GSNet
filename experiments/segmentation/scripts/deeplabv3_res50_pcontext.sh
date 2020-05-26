# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model deeplabv3 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname deeplabv3_res50_pcontext

#test [single-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model deeplabv3 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/deeplabv3/deeplabv3_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model deeplabv3 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/deeplabv3/deeplabv3_res50_pcontext/model_best.pth.tar --split val --mode testval --ms