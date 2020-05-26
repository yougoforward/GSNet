# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model gsnet --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname gsnet_res50_pcontext

#test [single-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model gsnet --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/gsnet/gsnet_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model gsnet --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/gsnet/gsnet_res50_pcontext/model_best.pth.tar --split val --mode testval --ms