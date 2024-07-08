#!/bin/bash

python3 /data/wb_project/SGDepth-master/train.py \
        --experiment-class sgdepth_eccv_test \
        --model-name kitti_full \
        --masking-enable \
        --masking-from-epoch 15 \
        --masking-linear-increase \
        --model-load /data/wb_project/SGDepth-master/my_models/sgdepth_eccv_test/kitti_full/checkpoints/epoch_10/