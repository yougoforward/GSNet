# !/usr/bin/env bash
# train
python train.py --dataset pcontext \
    --model gsnet_noatt --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname gsnet_noatt_res101_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model gsnet_noatt --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume runs/pcontext/gsnet_noatt/gsnet_noatt_res101_pcontext/checkpoint.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model gsnet_noatt --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume runs/pcontext/gsnet_noatt/gsnet_noatt_res101_pcontext/checkpoint.pth.tar --split val --mode testval --ms