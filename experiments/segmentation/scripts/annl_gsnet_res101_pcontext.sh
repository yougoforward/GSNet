# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model annl_gsnet --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname annl_gsnet_res101_pcontext

#test [single-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model annl_gsnet --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/annl_gsnet/annl_gsnet_res101_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model annl_gsnet --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/annl_gsnet/annl_gsnet_res101_pcontext/model_best.pth.tar --split val --mode testval --ms