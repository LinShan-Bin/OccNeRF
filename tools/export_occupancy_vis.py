import os
import argparse

import torch
import numpy as np
import mayavi.mlab as mlab


colors = np.array(
    [
        [0, 0, 0, 255],
        [255, 120, 50, 255],  # barrier              orangey
        [255, 192, 203, 255],  # bicycle              pink
        [255, 255, 0, 255],  # bus                  yellow
        [0, 150, 245, 255],  # car                  blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [200, 180, 0, 255],  # motorcycle           dark orange
        [255, 0, 0, 255],  # pedestrian           red
        [255, 240, 150, 255],  # traffic_cone         light yellow
        [135, 60, 0, 255],  # trailer              brown
        [160, 32, 240, 255],  # truck                purple
        [255, 0, 255, 255],  # driveable_surface    dark pink
        # [175,   0,  75, 255],       # other_flat           dark red
        [139, 137, 137, 255],
        [75, 0, 75, 255],  # sidewalk             dard purple
        [150, 240, 80, 255],  # terrain              light green
        [230, 230, 250, 255],  # manmade              white
        [0, 175, 0, 255],  # vegetation           green
        [0, 255, 127, 255],  # ego car              dark cyan
        [255, 99, 71, 255],
        [0, 191, 255, 255]
    ]
).astype(np.uint8)

threshold = 0.65
voxel_size = 0.4
pc_range = [-40, -40, -1, 40, 40, 5.4]
X, Y, Z = 200, 200, 16


def gridcloud3d(B, Z, Y, X, device='cpu'):
    # we want to sample for each location in the grid
    grid_z, grid_y, grid_x = meshgrid3d(B, Z, Y, X, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])

    # pdb.set_trace()
    # these are B x N
    xyz = torch.stack([x, y, z], dim=2)
    # here is stack in order with xyz
    # this is B x N x 3

    # pdb.set_trace()
    return xyz


def meshgrid3d(B, Z, Y, X, stack=False, device='cuda'):
    # returns a meshgrid sized B x Z x Y x X

    grid_z = torch.linspace(0.0, Z-1, Z, device=device)
    grid_z = torch.reshape(grid_z, [1, Z, 1, 1])
    grid_z = grid_z.repeat(B, 1, Y, X)

    grid_y = torch.linspace(0.0, Y-1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
    grid_y = grid_y.repeat(B, Z, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Z, Y, 1)
    # here repeat is in the order with ZYX

    if stack:
        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x


def visualize_occ_dict(output_dict, offscreen=True, render_w=1600):
    mlab.options.offscreen = offscreen

    data_dict = np.load(output_dict, allow_pickle=True).item()
    probs = data_dict['probability'][0]
    C, X_org, Y_org, Z_org = probs.shape
    x_offset, y_offset, z_offset = (X_org - X) // 2, (Y_org - Y) // 2, (Z_org - Z) // 2
    probs = probs[:, x_offset:x_offset+X, y_offset:y_offset+Y, z_offset:z_offset+Z]

    xyz = gridcloud3d(1, Z, Y, X, device='cpu')
    xyz_min = np.array(pc_range[:3])
    xyz_max = np.array(pc_range[3:])
    occ_size = np.array([X, Y, Z])
    xyz = xyz / occ_size * (xyz_max - xyz_min) + xyz_min + 0.5 * voxel_size
    xyz = xyz.reshape(Z, Y, X, 3).permute(2, 1, 0, 3).numpy()

    if C == 1:
        occs = 1 / (1 + np.exp(-probs[0]))
        occ_mask = (occs > threshold).astype(bool)
        xyz_out = xyz[occ_mask]
        xyz_class = np.concatenate([xyz_out, np.ones((xyz_out.shape[0], 1)) * 15], axis=1)
    else:
        occs = probs.argmax(axis=0)
        occ_mask = occs != C - 1
        xyz_out = xyz[occ_mask]
        xyz_class = np.concatenate([xyz_out, occs[occ_mask][:, None]], axis=1)


    fov_voxels = xyz_class

    cam_positions, focal_positions = [], []
    cam2ego_rts = data_dict['pose_spatial']
    f = 0.0055  # only define the direction
    for i in range(6):
        cam_position = cam2ego_rts[i] @ np.array([0., 0., 0., 1.]).reshape([4, 1])
        cam_positions.append(cam_position.flatten()[:3])
        focal_position = cam2ego_rts[i] @ np.array([0., 0., f, 1.]).reshape([4, 1])
        focal_positions.append(focal_position.flatten()[:3])
    cam_type = ['front', 'front_left', 'back_left', 'back', 'back_right', 'front_right']

    for i in [0, 1, 2, 3, 4, 5]:  # the first one will be broken, so we repeat it
        # figure = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
        figure = mlab.figure(size=(render_w, render_w/16*9), bgcolor=(1, 1, 1))
        # pdb.set_trace()
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            colormap="viridis",
            scale_factor=voxel_size - 0.05 * voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=0,
            vmax=19,
        )
        
        plt_plot_fov.glyph.scale_mode = "scale_by_vector"
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

        scene = figure.scene
        if i < 6:
            position = cam_positions[i]
            focal_position = focal_positions[i]
            print(f"position: {position}, focal_position: {focal_position}")
            scene.camera.position = position
            scene.camera.focal_point = focal_position
            scene.camera.view_angle = 35 if i != 3 else 60
            scene.camera.view_up = [0.0, 0.0, 1.0]
            scene.camera.clipping_range = [0.01, 300.]
            scene.camera.compute_view_plane_normal()
            scene.render()
        elif i == 6:
            # scene.camera.position = [-4.69302904, -52.74874688, 19.16181492]
            # scene.camera.focal_point = [-4.52985313, -51.8233303, 18.81979477]
            # scene.camera.view_angle = 40.0
            # scene.camera.view_up = [0.0, 0.0, 1.0]
            # scene.camera.clipping_range = [0.01, 300.]
            # scene.camera.compute_view_plane_normal()
            # scene.render()
            scene.camera.position = [  0.75131739, -35.08337438,  16.71378558]
            scene.camera.focal_point = [  0.75131739, -34.21734897,  16.21378558]
            scene.camera.view_angle = 40.0
            scene.camera.view_up = [0.0, 0.0, 1.0]
            scene.camera.clipping_range = [0.01, 300.]
            scene.camera.compute_view_plane_normal()
            scene.render()

        else:
            # scene.camera.position = [91.84365261779985, 87.2356528161641, 86.90232146965226]
            # scene.camera.focal_point = [4.607997894287109, -1.9073486328125e-06, -0.33333325386047363]
            # scene.camera.view_angle = 30.0
            # scene.camera.view_up = [0.0, 0.0, 1.0]
            # scene.camera.clipping_range = [33.458354318473965, 299.5433372220855]
            # scene.camera.compute_view_plane_normal()
            # scene.render()
            scene.camera.position = [ 0.75131739,  0.78265103, 93.21378558]
            scene.camera.focal_point = [ 0.75131739,  0.78265103, 92.21378558]
            scene.camera.view_angle = 40.0
            scene.camera.view_up = [0., 1., 0.]
            scene.camera.clipping_range = [0.01, 400.]
            scene.camera.compute_view_plane_normal()
            scene.render()

        save_path = output_dict.replace('visual_feature', 'visual_rgb_depth')
        mlab.show()
        mlab.savefig(f'{save_path}-{cam_type[i]}.jpg')
        if offscreen:
            mlab.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the Occupancy Grid')
    parser.add_argument('dict_folder', type=str, default='logs/nusc-sem-weight-sw16e-2-old/visual_feature')
    parser.add_argument('--offscreen', action='store_true')
    args = parser.parse_args()
    dict_folder = args.dict_folder
    dict_list = os.listdir(dict_folder)
    dict_list.sort()
    for dict_name in dict_list:
        if dict_name.endswith('.npy'):
            visualize_occ_dict(os.path.join(dict_folder, dict_name), offscreen=args.offscreen)
