# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import pdb
import pickle

import cv2
import torch
import numpy as np
import PIL.Image as pil
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from .mono_dataset import MonoDataset


class NuscDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(NuscDataset, self).__init__(*args, **kwargs)

        self.split = 'train' if self.is_train else 'val'
        self.data_path = os.path.join(self.opt.dataroot, 'nuscenes')
        if (not self.self_supervise) or (not self.is_train):
            self.depth_path = os.path.join(self.opt.dataroot, 'nuscenes_depth')
        if self.self_supervise and self.is_train and self.opt.use_semantic:
            self.semantic_path = os.path.join(self.opt.dataroot, 'nuscenes_semantic')
            self.semantic_map = np.array([
                0,   # ignore
                4,   # sedan      -> car
                11,  # highway    -> driveable_surface
                3,   # bus        -> bus
                10,  # truck      -> truck
                14,  # terrain    -> terrain
                16,  # tree       -> vegetation
                13,  # sidewalk   -> sidewalk
                2,   # bicycle    -> bycycle
                1,   # barrier    -> barrier
                7,   # person     -> pedestrian
                15,  # building   -> manmade
                6,   # motorcycle -> motorcycle
                5,   # crane      -> construction_vehicle
                9,   # trailer    -> trailer
                8,   # cone       -> traffic_cone
                17   # sky        -> ignore
            ], dtype=np.int8)

        if self.split == 'val':
            self.gts_path = os.path.join(self.opt.dataroot, 'gts')
        if not self.self_supervise:
            raise NotImplementedError

        version = 'v1.0-trainval'

        if 'nusc' in kwargs:
            self.nusc = kwargs['nusc']
        else:
            self.nusc = NuScenes(version=version, dataroot=self.data_path, verbose=False)


        if self.opt.data_type == 'all':
            with open('datasets/nusc/{}.txt'.format(self.split), 'r') as f:
                self.filenames = f.readlines()

        else:
            print('please define data type!!')
            exit()


        self.camera_ids_list = ['front', 'front_left', 'back_left', 'back', 'back_right', 'front_right',
                           'front', 'front_left', 'back_left', 'back', 'back_right']

        self.camera_names_list = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
                             'CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    def get_info(self, inputs, index_temporal, do_flip):
        inputs[("color", 0, -1)] = []

        if self.is_train:
            for idx, i in enumerate(self.frame_idxs[1:]):
                inputs[("color", i, -1)] = []
                inputs[("pose_spatial", i)] = []
                if self.volume_depth:
                    inputs[("cam_T_cam", i)] = []

            for idx, i in enumerate(self.frame_idxs):
                inputs[('K_ori', i)] = [] 

            inputs["pose_spatial"] = []
            
            if self.opt.use_semantic:
                inputs['semantic'] = []

        else:
            inputs[('K_ori', 0)] = [] 
            inputs['depth'] = []

            if self.opt.volume_depth:
                inputs["pose_spatial"] = []

        inputs['width_ori'], inputs['height_ori'], inputs['id'] = [], [], []

        rec = self.nusc.get('sample', index_temporal)
        if self.split == 'val' and self.opt.eval_occ:
            scene_name = self.nusc.get('scene', rec['scene_token'])['name']
            label_path = os.path.join(self.gts_path, scene_name, index_temporal, 'labels.npz')
            label = np.load(label_path)
            semantics_3d, mask_camera_3d, mask_lidar_3d = label['semantics'], label['mask_camera'], label['mask_lidar']
            inputs['semantics_3d'], inputs['mask_camera_3d'] = semantics_3d, np.logical_and(mask_camera_3d, mask_lidar_3d)

        camera_ids = ['front', 'front_left', 'back_left', 'back', 'back_right', 'front_right']
        camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
        
        ego_c = self.nusc.get('ego_pose', rec['data']['CAM_FRONT'])  # ego pose for the center frame
        egoc_to_global = rt2mat(ego_c['translation'], ego_c['rotation'])
        global_to_egoc = rt2mat(ego_c['translation'], ego_c['rotation'], inverse=True)
        
        # if render auxiliary frames
        cam_num = 6 * 3 if (self.opt.auxiliary_frame and self.is_train) else 6
        for cam_idx in range(cam_num):
            index_spatial, index_temporal = cam_idx % 6, cam_idx // 6
            
            cam_sample = self.nusc.get('sample_data', rec['data'][camera_names[index_spatial]])
            if index_temporal == 1:  # get previous frame for auxiliary supervision
                cam_sample = self.nusc.get('sample_data', cam_sample['prev'])
                if len(cam_sample['prev']) == 0:
                    continue
                camera_id = camera_ids[index_spatial] + '_prev'
            elif index_temporal == 2:  # get next frame for auxiliary supervision
                cam_sample = self.nusc.get('sample_data', cam_sample['next'])
                if len(cam_sample['next']) == 0:
                    continue
                camera_id = camera_ids[index_spatial] + '_next'
            else:
                camera_id = camera_ids[index_spatial]

            inputs['id'].append(camera_id)

            ego0 = self.nusc.get('ego_pose', cam_sample['token'])
            ego0_to_global = rt2mat(ego0['translation'], ego0['rotation'])
            global_to_ego0 = rt2mat(ego0['translation'], ego0['rotation'], inverse=True)

            color = self.loader(os.path.join(self.data_path, cam_sample['filename']))
            inputs['width_ori'].append(color.size[0])
            inputs['height_ori'].append(color.size[1])

            if not self.is_train and index_temporal == 0:
                depth = np.load(os.path.join(self.depth_path, cam_sample['filename'][:-4] + '.npy'))
                inputs['depth'].append(depth.astype(np.float32))

            """Add:
            semantic  # TODO: optional, add semantic for auxiliary frame
            """
            if self.self_supervise and self.is_train and self.opt.use_semantic:
                semantic = np.fromfile(os.path.join(self.semantic_path, cam_sample['filename'][:-4] + '_mask.bin'), dtype=np.int8).reshape(900, 1600)
                semantic = self.semantic_map[semantic]
                inputs['semantic'].append(semantic)
            
            if do_flip:
                color = color.transpose(pil.FLIP_LEFT_RIGHT)
            inputs[("color", 0, -1)].append(color)

            '''
            Definition of a particular sensor (lidar/radar/camera) as calibrated on a particular vehicle. 
            All extrinsic parameters are given with respect to the ego vehicle body frame. All camera images come undistorted and rectified.
            '''
            ego_spatial = self.nusc.get('calibrated_sensor', cam_sample['calibrated_sensor_token'])
            cam0_to_ego0 = rt2mat(ego_spatial['translation'], ego_spatial['rotation'])

            # For auxiliary frame, we need to convert the pose to the center frame
            if index_temporal != 0:
                cam0_to_egoc = global_to_egoc @ ego0_to_global @ cam0_to_ego0
            else:
                cam0_to_egoc = cam0_to_ego0

            if self.is_train or self.opt.volume_depth:
                inputs["pose_spatial"].append(cam0_to_egoc.astype(np.float32))

            K = np.eye(4).astype(np.float32)
            K[:3, :3] = ego_spatial['camera_intrinsic']
            inputs[('K_ori', 0)].append(K)

            if self.is_train:

                for idx, i in enumerate(self.frame_idxs[1:]):
                    if i == -1:
                        index_temporal_i = cam_sample['prev']
                    elif i == 1:
                        index_temporal_i = cam_sample['next']
                    cam_sample_i = self.nusc.get(
                        'sample_data', index_temporal_i)
                    ego_spatial_i = self.nusc.get(
                        'calibrated_sensor', cam_sample_i['calibrated_sensor_token'])

                    K = np.eye(4).astype(np.float32)
                    K[:3, :3] = ego_spatial_i['camera_intrinsic']

                    inputs[('K_ori', i)].append(K)

                    color_i = self.loader(os.path.join(self.data_path, cam_sample_i['filename']))

                    if do_flip:
                        color_i = color_i.transpose(pil.FLIP_LEFT_RIGHT)

                    inputs[("color", i, -1)].append(color_i)
                    
                    if self.volume_depth:
                        # Get ground truth camera to camera transformation
                        egoi = self.nusc.get('ego_pose', cam_sample_i['token'])
                        global_to_egoi = rt2mat(egoi['translation'], egoi['rotation'], inverse=True)
                        egoi_to_cami = rt2mat(ego_spatial_i['translation'], ego_spatial_i['rotation'], inverse=True)
                        cam0_to_cami = egoi_to_cami @ global_to_egoi @ ego0_to_global @ cam0_to_ego0
                        inputs['cam_T_cam', i].append(cam0_to_cami)


        if self.is_train or self.opt.volume_depth:

            if self.is_train:
                for index_spatial in range(6):
                    for i in [-1, 1]:
                        pose_0_spatial = inputs["pose_spatial"][index_spatial]
                        pose_i_spatial = inputs["pose_spatial"][(index_spatial+i)%6] # 围绕一个转动方向取相邻相机的RT：

                        gt_pose_spatial = np.linalg.inv(pose_i_spatial) @ pose_0_spatial
                        inputs[("pose_spatial", i)].append(gt_pose_spatial.astype(np.float32))

                if self.volume_depth:
                    for frame_idx in self.frame_idxs[1:]:
                        inputs['cam_T_cam', frame_idx] = np.stack(inputs['cam_T_cam', frame_idx], axis=0)

            for idx, i in enumerate(self.frame_idxs):
                if self.is_train:
                    inputs[('K_ori', i)] = np.stack(inputs[('K_ori', i)], axis=0)

                    if i != 0:
                        inputs[("pose_spatial", i)] = np.stack(inputs[("pose_spatial", i)], axis=0)
                        # stack the RT in the same time
                else:
                    inputs[('K_ori', 0)] = np.stack(inputs[('K_ori', 0)], axis=0)

            if 'depth' in inputs.keys():
                inputs['depth'] = np.stack(inputs['depth'], axis=0)

            inputs['pose_spatial'] = np.stack(inputs['pose_spatial'], axis=0)

            if 'semantic' in inputs.keys():
                inputs['semantic'] = np.stack(inputs['semantic'], axis=0)

        else:
            inputs[('K_ori', 0)] = np.stack(inputs[('K_ori', 0)], axis=0) 
            if 'depth' in inputs.keys():
                inputs['depth'] = np.stack(inputs['depth'], axis=0)

            if self.opt.volume_depth:
                inputs['pose_spatial'] = np.stack(inputs['pose_spatial'], axis=0)


        for key in ['width_ori', 'height_ori']:
            inputs[key] = np.stack(inputs[key], axis=0)

        return


def rt2mat(translation, quaternion=None, inverse=False, rotation=None):
    R = Quaternion(quaternion).rotation_matrix if rotation is None else rotation
    T = np.array(translation)
    if inverse:
        R = R.T
        T = -R @ T
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T
    return mat

def split_rt_single(rt):
    r = rt[:3, :3]
    t = rt[:3, 3].view(3)
    return r, t

def safe_inverse_single(a):
    r, t = split_rt_single(a)
    t = t.view(3,1)
    r_transpose = r.t()
    inv = torch.cat([r_transpose, -torch.matmul(r_transpose, t)], 1)
    bottom_row = a[3:4, :] # this is [0, 0, 0, 1]
    # bottom_row = torch.tensor([0.,0.,0.,1.]).view(1,4)
    inv = torch.cat([inv, bottom_row], 0)
    return inv


def safe_inverse(a): #parallel version
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1,2) #inverse of rotation matrix

    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])

    return inv
