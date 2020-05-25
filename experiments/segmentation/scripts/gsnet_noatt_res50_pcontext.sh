# !/usr/bin/env bash
# train
python train.py --dataset pcontext \
    --model gsnet_noatt --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname gsnet_noatt_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model gsnet_noatt --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume runs/pcontext/gsnet_noatt/gsnet_noatt_res50_pcontext/checkpoint.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model gsnet_noatt --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume runs/pcontext/gsnet_noatt/gsnet_noatt_res50_pcontext/checkpoint.pth.tar --split val --mode testval --ms