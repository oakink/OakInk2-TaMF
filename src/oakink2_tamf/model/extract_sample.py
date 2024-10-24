import torch

from ..dataset.collate import interaction_segment_collate
from dev_fn.transform.cast import map_copy_select_to


def extract_refined_sample(generation_model, diffusion, refine_model, gt_sample, device, dtype):
    batch = interaction_segment_collate([gt_sample])
    batch_device = map_copy_select_to(
        batch,
        device=device,
        dtype=dtype,
        select=("mask", "pose_repr", "shape", "obj_num", "obj_traj", "obj_embedding"),
    )
    with torch.no_grad():
        sample_fn = diffusion.p_sample_loop
        generation_model.eval()
        input_shape = tuple(batch_device["pose_repr"].shape)
        input_shape = (input_shape[0], input_shape[2], 1, input_shape[1])
        sample = sample_fn(
            generation_model,
            input_shape,  # adapt from mdm
            clip_denoised=False,
            model_kwargs={"batch": batch_device},
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
    sample_ = sample.permute((0, 3, 1, 2)).squeeze(3)
    batch_device["sample_pose_repr"] = sample_
    sample_np = sample_.detach().clone().cpu().numpy().squeeze(0)

    with torch.no_grad():
        refine_model.eval()
        output = refine_model(batch_device)

    sample_np = output["refine_pose_repr"].detach().clone().cpu().numpy().squeeze(0)
    return sample_np


def extract_refined_sample_bihand(generation_model, diffusion, refine_model, gt_sample, hand_side, device, dtype):
    text = gt_sample["text"]
    avai_len = gt_sample["len"]
    # hand_side = gt_sample["hand_side"]
    pose_repr_lh = gt_sample["pose_repr_lh"]
    shape_lh = gt_sample["shape_lh"]
    pose_repr_rh = gt_sample["pose_repr_rh"]
    shape_rh = gt_sample["shape_rh"]
    obj_list = gt_sample["obj_list"]
    obj_traj = gt_sample["obj_traj"]
    obj_embedding = gt_sample["obj_embedding"]
    obj_pointcloud = gt_sample["obj_pointcloud"]
    obj_pair = gt_sample["obj_pair"]

    magic_index = 1 if hand_side == "rh" else 0
    pose_repr = pose_repr_rh if hand_side == "rh" else pose_repr_lh
    shape = shape_rh if hand_side == "rh" else shape_lh

    # format rh sample
    oid_indices = [obj_list.index(_oid) for _oid in obj_pair[magic_index]]
    sample = {
        "text": text,
        "len": avai_len,
        "mask": gt_sample["mask"],
        "hand_side": hand_side,
        "pose_repr": pose_repr,
        "shape": shape,
        "obj_num": len(obj_pair[magic_index]),
        "obj_list": obj_pair[magic_index],
        "obj_traj": obj_traj[oid_indices, ...],
        "obj_embedding": obj_embedding[oid_indices, ...],
        "obj_pointcloud": obj_pointcloud[oid_indices, ...],
    }
    batch = interaction_segment_collate([sample])
    batch_device = map_copy_select_to(
        batch,
        device=device,
        dtype=dtype,
        select=("mask", "pose_repr", "shape", "obj_num", "obj_traj", "obj_embedding"),
    )
    with torch.no_grad():
        sample_fn = diffusion.p_sample_loop
        generation_model.eval()
        input_shape = tuple(batch_device["pose_repr"].shape)
        input_shape = (input_shape[0], input_shape[2], 1, input_shape[1])
        sample = sample_fn(
            generation_model,
            input_shape,  # adapt from mdm
            clip_denoised=False,
            model_kwargs={"batch": batch_device},
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
    sample_ = sample.permute((0, 3, 1, 2)).squeeze(3)
    batch_device["sample_pose_repr"] = sample_
    sample_np = sample_.detach().clone().cpu().numpy().squeeze(0)

    with torch.no_grad():
        refine_model.eval()
        output = refine_model(batch_device)

    sample_np = output["refine_pose_repr"].detach().clone().cpu().numpy().squeeze(0)
    return sample_np