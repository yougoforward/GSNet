# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset cocostuff \
    --model new_psp3 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname new_psp3_res101_cocostuff

#test [single-scale]
python -m experiments.segmentation.test --dataset cocostuff \
    --model new_psp3 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/cocostuff/new_psp3/new_psp3_res101_cocostuff/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset cocostuff \
    --model new_psp3 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/cocostuff/new_psp3/new_psp3_res101_cocostuff/model_best.pth.tar --split val --mode testval --ms