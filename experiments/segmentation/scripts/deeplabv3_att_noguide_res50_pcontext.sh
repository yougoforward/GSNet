# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model deeplabv3_att_noguide --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname deeplabv3_att_noguide_res50_pcontext

#test [single-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model deeplabv3_att_noguide --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/deeplabv3_att_noguide/deeplabv3_att_noguide_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model deeplabv3_att_noguide --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/deeplabv3_att_noguide/deeplabv3_att_noguide_res50_pcontext/model_best.pth.tar --split val --mode testval --ms