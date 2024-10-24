#!/bin/bash

python -m oakink2_tamf.launch.train \
    --cfg config/data_reverse_segment.yml \
    --cfg config/obj_embedding.yml \
    --cfg config/obj_pointcloud.yml \
    --cfg config/cache_dict.yml \
    --cfg config/arch_mdm_l.yml \
    --cfg config/loss_param.yml \
    --cfg config/bs_64.yml \
    --runtime.device_id 0,1,2,3 \
    --val.val_freq -1 --test.test_freq -1 \
    --exp_id "main__?(ts)" \
    --commit
