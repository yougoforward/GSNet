# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model new_psp3_aspp_base_psaa_att --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname new_psp3_aspp_base_psaa_att_res101_pcontext

#test [single-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model new_psp3_aspp_base_psaa_att --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/new_psp3_aspp_base_psaa_att/new_psp3_aspp_base_psaa_att_res101_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model new_psp3_aspp_base_psaa_att --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/new_psp3_aspp_base_psaa_att/new_psp3_aspp_base_psaa_att_res101_pcontext/model_best.pth.tar --split val --mode testval --ms