#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

./tools/test_image_with_ms_transformer.py  \
--imgdir data/demo   \
--color *-color.png   \
--depth *-depth.png \
--cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
--pretrained data/checkpoints/norm_RGBD_finetuned_data04_OCID_5epoch.pth \
--pretrained_crop data/checkpoints/crop_RGBD_finetuned_data04_OSD_1epoch.pth \
--network_cfg MSMFormer/configs/mixture_UCN.yaml \
--network_crop_cfg MSMFormer/configs/crop_mixture_UCN.yaml \
--input_image RGBD_ADD
