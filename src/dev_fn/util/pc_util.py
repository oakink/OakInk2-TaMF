import numpy as np


def rgbd_to_pc(color_img, depth_img, cam_intr, cam_extr=None, z_max=3.0, depth_scale=1000.0):
    color_img = np.asarray(color_img)
    depth_img = np.asarray(depth_img)

    img_shape = color_img.shape
    img_h, img_w, _ = img_shape

    coord_y = np.arange(img_h)
    coord_x = np.arange(img_w)
    coord_xy = np.stack(np.meshgrid(coord_x, coord_y), axis=-1)  # (H, W, 2)

    xy_in = coord_xy.reshape(-1, 2)  # [NRAW, 2]
    z_in = depth_img.reshape(-1, 1) / depth_scale  # [NRAW, 1]
    if z_max is not None:
        z_filter = np.nonzero(np.logical_and(z_in.ravel() > 0, z_in.ravel() < z_max))

    f = cam_intr[(0, 1), (0, 1)].reshape((1, 2))
    pp = cam_intr[0:2, 2].reshape((1, 2))

    xy = xy_in[z_filter]  # [N, 2]
    z = z_in[z_filter]  # [N, 1]
    xy_ = (xy - pp) / f * z

    xyz = np.concatenate((xy_, z), axis=1)
    rgb = color_img.reshape(-1, 3)[z_filter]
    rgb = rgb[:, (2, 1, 0)] / 255

    if cam_extr is not None:
        # revert cam_extr
        R_extr_inv = cam_extr[:3, :3].T
        t_extr_inv = -R_extr_inv @ cam_extr[:3, 3]
        xyz = (R_extr_inv @ xyz.T).T + t_extr_inv.reshape((1, 3))

    return xyz, rgb
