#!/usr/bin/env bash
#train
python -m experiments.segmentation.train_can --dataset pcontext \
    --model can --aux --dilated --multi-grid --stride 16 --atrous-rates 6 12 18 --base-size 520 --crop-size 520 --batch-size 6 \
    --backbone resnet50 --checkname can_res50_pcontext

#test [single-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model can --aux --dilated --multi-grid --stride 16 --atrous-rates 6 12 18 --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/can/can_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model can --aux --dilated --multi-grid --stride 16 --atrous-rates 6 12 18 --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/can/can_res50_pcontext/model_best.pth.tar --split val --mode testval --ms