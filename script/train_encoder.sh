#!/bin/bash

python -m oakink2_tamf.launch.train_encoder \
    --cfg config/obj_embedding.yml \
    --cfg config/obj_pointcloud.yml \
    --cfg config/cache_dict.yml \
    --cfg config/refine_sample_param.yml \
    --cfg config/arch_encoder.yml \
    --cfg config/bs_256.yml \
    --train.num_epoch 400 \
    --train.scheduler_milestone 80,160,240,320 \
    --runtime.num_worker 0 \
    --runtime.device_id 0,1,2,3 \
    --val.val_freq 20 --test.test_freq 20 \
    --exp_id "encoder__?(ts)" \
    --commit

