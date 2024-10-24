#!/bin/bash

python -m oakink2_tamf.launch.train_refine \
    --cfg config/obj_embedding.yml \
    --cfg config/obj_pointcloud.yml \
    --cfg config/cache_dict.yml \
    --cfg config/refine_sample_param.yml \
    --cfg config/arch_mdm.yml \
    --cfg config/loss_param_refine.yml \
    --cfg config/bs_64.yml \
    --runtime.device_id 0,1,2,3 \
    --val.val_freq 20 --test.test_freq 40 \
    --exp_id "refine__?(ts)" \
    --commit

