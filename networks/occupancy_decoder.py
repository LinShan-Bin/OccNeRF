# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import pdb
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_efficient_distloss import eff_distloss, eff_distloss_native

from utils import geom
from utils import vox
from utils import basic
from utils import render
from ._3DCNN import S3DCNN


class VolumeDecoder(nn.Module):

    def __init__(self, opt):
        super(VolumeDecoder, self).__init__()

        self.opt = opt
        self.use_semantic = self.opt.use_semantic
        self.semantic_classes = self.opt.semantic_classes
        self.batch = self.opt.batch_size // self.opt.cam_N

        self.near = self.opt.min_depth
        self.far = self.opt.max_depth

        self.register_buffer('xyz_min', torch.from_numpy(
            np.array([self.opt.real_size[0], self.opt.real_size[2], self.opt.real_size[4]])))
        self.register_buffer('xyz_max', torch.from_numpy(
            np.array([self.opt.real_size[1], self.opt.real_size[3], self.opt.real_size[5]])))

        self.ZMAX = self.opt.real_size[1]

        self.Z = self.opt.voxels_size[0]
        self.Y = self.opt.voxels_size[1]
        self.X = self.opt.voxels_size[2]

        self.Z_final = self.Z
        self.Y_final = self.Y
        self.X_final = self.X


        self.stepsize = self.opt.stepsize  # voxel
        self.num_voxels = self.Z_final * self.Y_final * self.X_final
        self.stepsize_log = self.stepsize
        self.interval = self.stepsize

        if self.opt.contracted_coord:
            # Sampling strategy for contracted coordinate
            contracted_rate = self.opt.contracted_ratio
            num_id_voxels = int(self.num_voxels * (contracted_rate)**3)
            self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_id_voxels).pow(1 / 3)
            diagonal = (self.xyz_max - self.xyz_min).pow(2).sum().pow(1 / 2)
            self.N_samples = int(diagonal / 2 / self.stepsize / self.voxel_size / contracted_rate)
            if self.opt.infinite_range:
                # depth_roi = [-self.far] * 3 + [self.far] * 3
                zval_roi = [-diagonal] * 3 + [diagonal] * 3
                fc = 1 - 0.5 / self.X  # avoid NaN
                zs_contracted = torch.linspace(0.0, fc, steps=self.N_samples)
                zs_world = vox.contracted2world(
                    zs_contracted[None, :, None].repeat(1, 1, 3),
                    # pc_range_roi=depth_roi,
                    pc_range_roi=zval_roi,
                    ratio=self.opt.contracted_ratio)[:, :, 0]
            else:
                zs_world = torch.linspace(0.0, self.N_samples - 1, steps=self.N_samples)[None] * self.stepsize * self.voxel_size
            self.register_buffer('Zval', zs_world)

            pc_range_roi = self.xyz_min.tolist() + self.xyz_max.tolist()
            self.norm_func = lambda xyz: vox.world2contracted(xyz, pc_range_roi=pc_range_roi, ratio=self.opt.contracted_ratio)

        else:
            self.N_samples = int(np.linalg.norm(np.array([self.Z_final // 2, self.Y_final // 2, self.X_final // 2]) + 1) / self.stepsize) + 1
            self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels).pow(1 / 3)
            zs_world = torch.linspace(0.0, self.N_samples - 1, steps=self.N_samples)[None] * self.stepsize * self.voxel_size
            self.register_buffer('Zval', zs_world)
            self.norm_func = lambda xyz: (xyz - self.xyz_min.to(xyz)) / (self.xyz_max.to(xyz) - self.xyz_min.to(xyz)) * 2.0 - 1.0

        length_pose_encoding = 3


        if self.opt.position == 'embedding':
            input_channel = self.opt.input_channel
            self.pos_embedding = torch.nn.Parameter(torch.ones(
                [1, input_channel, self.opt.voxels_size[1], self.opt.voxels_size[2], self.opt.voxels_size[0]]))

        elif self.opt.position == 'embedding1':
            input_channel = self.opt.input_channel
            xyz_in_channels = 1 + 3

            embedding_width = 192
            embedding_depth = 5

            self.embeddingnet = nn.Sequential(
                nn.Linear(xyz_in_channels, embedding_width), nn.ReLU(inplace=True),
                *[nn.Sequential(nn.Linear(embedding_width, embedding_width), nn.ReLU(inplace=True))
                    for _ in range(embedding_depth - 2)], nn.Linear(embedding_width, self.opt.input_channel),)

            nn.init.constant_(self.embeddingnet[-1].bias, 0)
            self.pos_embedding1 = None
            self.pos_embedding_save = torch.nn.Parameter(torch.zeros([1, input_channel, self.opt.voxels_size[1], self.opt.voxels_size[2], self.opt.voxels_size[0]]), requires_grad= False)

        else:
            self.pos_embedding = None
            self.pos_embedding1 = None
            input_channel = self.opt.input_channel

        scene_centroid_x = 0.0
        scene_centroid_y = 0.0
        scene_centroid_z = 0.0

        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])

        self.register_buffer('scene_centroid', torch.from_numpy(scene_centroid).float())

        self.bounds = (self.opt.real_size[0], self.opt.real_size[1],
                       self.opt.real_size[2], self.opt.real_size[3],
                       self.opt.real_size[4], self.opt.real_size[5])
        #  bounds = (-40, 40, -40, 40, -1, 5.4)

        self.vox_util = vox.Vox_util(
            self.Z, self.Y, self.X,
            scene_centroid=self.scene_centroid,
            bounds=self.bounds, position = self.opt.position, length_pose_encoding = length_pose_encoding, opt = self.opt,
            assert_cube=False)

        if self.opt.position != 'No' and self.opt.position != 'embedding':
            self.meta_data = self.vox_util.get_meta_data(cam_center=torch.Tensor([[1.2475, 0.0673, 1.5356]]), camB_T_camA=None).to('cuda')


        activate_fun = nn.ReLU(inplace=True)
        if self.opt.aggregation == '3dcnn':
            out_channel = self.opt.out_channel
            self._3DCNN = S3DCNN(input_planes=input_channel, out_planes=out_channel, planes=self.opt.con_channel,
                                 activate_fun=activate_fun, opt=opt)
        else:
            print('please define the aggregation')
            exit()

    def feature2vox_simple(self, features, pix_T_cams, cam0_T_camXs, __p, __u):

        pix_T_cams_ = pix_T_cams
        camXs_T_cam0_ = geom.safe_inverse(cam0_T_camXs)

        _, C, Hf, Wf = features.shape

        sy = Hf / float(self.opt.height)
        sx = Wf / float(self.opt.width)

        # unproject image feature to 3d grid
        featpix_T_cams_ = geom.scale_intrinsics(pix_T_cams_, sx, sy)
        # pix_T_cams_ shape: [6,4,4]  feature down sample -> featpix_T_cams_

        feat_mems_ = self.vox_util.unproject_image_to_mem(
            features,
            basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, self.Z, self.Y, self.X)

        # feat_mems_ shape： torch.Size([6, 128, 200, 8, 200])
        feat_mems = __u(feat_mems_)  # B, S, C, Z, Y, X # torch.Size([1, 6, 128, 200, 8, 200])

        mask_mems = (torch.abs(feat_mems) > 0).float()
        feat_mem = basic.reduce_masked_mean(feat_mems, mask_mems, dim=1)  # B, C, Z, Y, X
        feat_mem = feat_mem.permute(0, 1, 4, 3, 2) # [0, ...].unsqueeze(0) # ZYX -> XYZ

        return feat_mem

    def grid_sampler(self, xyz, *grids, align_corners=True, avail_mask=None, vis=False):
        '''Wrapper for the interp operation'''
        # pdb.set_trace()
        
        if self.opt.semantic_sample_ratio < 1.0 and self.use_semantic and not vis:
            group_size = int(1.0 / self.opt.semantic_sample_ratio)
            group_num = xyz.shape[1] // group_size
            xyz_sem = xyz[:, :group_size * group_num].reshape(xyz.shape[0], group_num, group_size, 3).mean(dim=2)
        else:
            xyz_sem = None

        if avail_mask is not None:
            if self.opt.contracted_coord:
                ind_norm = self.norm_func(xyz)
                avail_mask = self.effective_points_mask(ind_norm)
                ind_norm = ind_norm[avail_mask]
                if xyz_sem is not None:
                    avail_mask_sem = avail_mask[:, :group_size * group_num].reshape(avail_mask.shape[0], group_num, group_size).any(dim=-1)
                    ind_norm_sem = self.norm_func(xyz_sem[avail_mask_sem])
            else:
                xyz_masked = xyz[avail_mask]
                ind_norm = self.norm_func(xyz_masked)
                if xyz_sem is not None:
                    avail_mask_sem = avail_mask[:, :group_size * group_num].reshape(avail_mask.shape[0], group_num, group_size).any(dim=-1)
                    ind_norm_sem = self.norm_func(xyz_sem[avail_mask_sem])
        else:
            ind_norm = self.norm_func(xyz)
            if xyz_sem is not None:
                ind_norm_sem = self.norm_func(xyz_sem)
                avail_mask_sem = None
        
        ind_norm = ind_norm.flip((-1,)) # value range: [-1, 1]
        shape = ind_norm.shape[:-1]
        ind_norm = ind_norm.reshape(1, 1, 1, -1, 3)
        if xyz_sem is None:
            grid = grids[0] # BCXYZ # torch.Size([1, C, 256, 256, 16])
            ret_lst = F.grid_sample(grid, ind_norm, mode='bilinear', align_corners=align_corners).reshape(grid.shape[1], -1).T.reshape(*shape, grid.shape[1])
            if self.use_semantic:
                semantic, feats = ret_lst[..., :self.semantic_classes], ret_lst[..., -1]
                return feats, avail_mask, semantic
            else:
                return ret_lst.squeeze(), avail_mask
        else:
            ind_norm_sem = ind_norm_sem.flip((-1,))
            shape_sem = ind_norm_sem.shape[:-1]
            ind_norm_sem = ind_norm_sem.reshape(1, 1, 1, -1, 3)
            grid_sem = grids[0][:, :self.semantic_classes] # BCXYZ # torch.Size([1, semantic_classes, H, W, Z])
            grid_geo = grids[0][:, -1:] # BCXYZ # torch.Size([1, 1, H, W, Z])
            ret_sem = F.grid_sample(grid_sem, ind_norm_sem, mode='bilinear', align_corners=align_corners).reshape(grid_sem.shape[1], -1).T.reshape(*shape_sem, grid_sem.shape[1])
            ret_geo = F.grid_sample(grid_geo, ind_norm, mode='bilinear', align_corners=align_corners).reshape(grid_geo.shape[1], -1).T.reshape(*shape, grid_geo.shape[1])
            return ret_geo.squeeze(), avail_mask, ret_sem, avail_mask_sem, group_num, group_size

    def sample_ray(self, rays_o, rays_d, is_train):
        '''Sample query points on rays'''
        Zval = self.Zval.to(rays_o)
        if is_train:
            Zval = Zval.repeat(rays_d.shape[-2], 1)
            Zval += (torch.rand_like(Zval[:, [0]]) * 0.2 - 0.1) * self.stepsize_log * self.voxel_size
            Zval = Zval.clamp(min=0.0)

        Zval = Zval + self.near
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * Zval[..., None]
        rays_pts_depth = (rays_o[..., None, :] - rays_pts).norm(dim=-1)

        if self.opt.contracted_coord:
            # contracted coordiante has infinite perception range
            mask_outbbox = torch.zeros_like(rays_pts[..., 0]).bool()
        else:
            mask_outbbox = ((self.xyz_min > rays_pts) | (rays_pts > self.xyz_max)).any(dim=-1)

        return rays_pts, mask_outbbox, Zval, rays_pts_depth
    
    def effective_points_mask(self, points):
        '''Mask out points that are too close to each other in the contracted coordinate'''
        dist = torch.diff(points, dim=-2, prepend=torch.zeros_like(points[..., :1, :])).abs()
        xyz_thresh = 0.4 / torch.tensor([self.X, self.Y, self.Z]).to(points)
        mask = (dist > xyz_thresh).bool().any(dim=-1)
        return mask

    def activate_density(self, density, dists):
        return 1 - torch.exp(-F.relu(density) * dists)
    
    def get_density(self, rays_o, rays_d, Voxel_feat, is_train, inputs):
        dtype = torch.float16 if self.opt.use_fp16 else torch.float32
        device = rays_o.device
        rays_o, rays_d, Voxel_feat = rays_o.to(dtype), rays_d.to(dtype), Voxel_feat.to(dtype)

        reg_loss = {}
        eps_time = time.time()
        with torch.no_grad():
            rays_o_i = rays_o[0, ...].flatten(0, 2)  # HXWX3
            rays_d_i = rays_d[0, ...].flatten(0, 2)  # HXWX3
            rays_pts, mask_outbbox, z_vals, rays_pts_depth = self.sample_ray(rays_o_i, rays_d_i, is_train=is_train)

        dists = rays_pts_depth[..., 1:] - rays_pts_depth[..., :-1]  # [num pixels, num points - 1]
        dists = torch.cat([dists, 1e4 * torch.ones_like(dists[..., :1])], dim=-1)  # [num pixels, num points]

        sample_ret = self.grid_sampler(rays_pts, Voxel_feat, avail_mask=~mask_outbbox)
        if self.use_semantic:
            if self.opt.semantic_sample_ratio < 1.0:
                geo_feats, mask, semantic, mask_sem, group_num, group_size = sample_ret
            else:
                geo_feats, mask, semantic = sample_ret
        else:
            geo_feats, mask = sample_ret


        if self.opt.render_type == 'prob':
            weights = torch.zeros_like(rays_pts[..., 0])
            weights[:, -1] = 1
            geo_feats = torch.sigmoid(geo_feats)
            if self.opt.last_free:
                geo_feats = 1.0 - geo_feats  # the last channel is the probability of being free
            weights[mask] = geo_feats

            # accumulate
            weights = weights.cumsum(dim=1).clamp(max=1)
            alphainv_fin = weights[..., -1]
            weights = weights.diff(dim=1, prepend=torch.zeros((rays_pts.shape[:1])).unsqueeze(1).to(device=device, dtype=dtype))
            depth = (weights * z_vals).sum(-1)
            rgb_marched = 0

        elif self.opt.render_type == 'density':
            alpha = torch.zeros_like(rays_pts[..., 0])  # [num pixels, num points]
            alpha[mask] = self.activate_density(geo_feats, dists[mask])

            weights, alphainv_cum = render.get_ray_marching_ray(alpha)
            alphainv_fin = alphainv_cum[..., -1]
            depth = (weights * z_vals).sum(-1)
            rgb_marched = 0

        else:
            raise NotImplementedError
        
        if self.use_semantic:
            if self.opt.semantic_sample_ratio < 1.0:
                semantic_out = torch.zeros(mask_sem.shape + (self.semantic_classes, )).to(device=device, dtype=dtype)
                semantic_out[mask_sem] = semantic
                weights_sem = weights[:, :group_num * group_size].reshape(weights.shape[0], group_num, group_size).sum(dim=-1)
                semantic_out = (semantic_out * weights_sem[..., None]).sum(dim=-2)
                
            else:
                semantic_out = torch.ones(rays_pts.shape[:-1] + (self.semantic_classes, )).to(device=device, dtype=dtype)
                semantic_out[mask] = semantic
                semantic_out = (semantic_out * weights[..., None]).sum(dim=-2)
        else:
            semantic_out = None

        if is_train:
            if self.opt.weight_entropy_last > 0:
                pout = alphainv_fin.float().clamp(1e-6, 1-1e-6)
                entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
                reg_loss["loss_entropy_last"] = self.opt.weight_entropy_last * entropy_last_loss

            if self.opt.weight_distortion > 0:
                loss_distortion = eff_distloss(weights.float(), z_vals.float(), dists.float())
                reg_loss['loss_distortion'] =  self.opt.weight_distortion * loss_distortion
            
            if self.opt.weight_sparse_reg > 0:
                geo_f = Voxel_feat[..., -1].float().flatten()
                if self.opt.last_free:
                    geo_f = - geo_f
                loss_sparse_reg = F.binary_cross_entropy_with_logits(geo_f, torch.zeros_like(geo_f), reduction='mean')
                reg_loss['loss_sparse_reg'] = self.opt.weight_sparse_reg * loss_sparse_reg

        return depth.float(), rgb_marched, semantic_out, reg_loss

    def forward(self, features, inputs, outputs={}, is_train=True, Voxel_feat_list=None, no_depth=False):

        __p = lambda x: basic.pack_seqdim(x, self.batch)  # merge batch and number of cameras
        __u = lambda x: basic.unpack_seqdim(x, self.batch)

        self.outputs = outputs

        if Voxel_feat_list is None:
            # 2D to 3D
            Voxel_feat= self.feature2vox_simple(features[0][:self.opt.cam_N], inputs[('K', 0, 0)][:self.opt.cam_N], inputs['pose_spatial'][:self.opt.cam_N], __p, __u)

            # position prior
            if self.opt.position == 'embedding':
                Voxel_feat = Voxel_feat * self.pos_embedding

            elif self.opt.position == 'embedding1':
                if is_train:
                    embedding = self.embeddingnet(self.meta_data)
                    embedding = torch.reshape(embedding, [self.opt.B, self.Z, self.Y, self.X, self.opt.input_channel]).permute(0, 4, 3, 2, 1)
                    self.pos_embedding_save.data = embedding

                else:
                    embedding = self.pos_embedding_save

                if self.opt.position == 'embedding1':
                    Voxel_feat = Voxel_feat * embedding

                else:
                    print('please define the opt.position')
                    exit()

            elif self.opt.position == 'No':
                pass

            else:
                print('please define the opt.position')
                exit()


            # 3D aggregation
            Voxel_feat_list = self._3DCNN(Voxel_feat)
        

        # rendering
        rays_o = __u(inputs['rays_o', 0])
        rays_d = __u(inputs['rays_d', 0])

        if is_train:
            for scale in self.opt.scales:
                cam_num = self.opt.cam_N * 3 if self.opt.auxiliary_frame else self.opt.cam_N
                depth, rgb_marched, semantic, reg_loss = self.get_density(rays_o, rays_d, Voxel_feat_list[scale], is_train, inputs)
                depth = depth.reshape(cam_num, self.opt.render_h, self.opt.render_w).unsqueeze(1)
                if self.opt.infinite_range:
                    depth = depth.clamp(min=self.near, max=200)  # for training stability
                else:
                    depth = depth.clamp(min=self.near, max=self.far)
                self.outputs[("disp", scale)] = depth
                if semantic is not None:
                    semantic = semantic.reshape(cam_num, self.opt.render_h, self.opt.render_w, self.semantic_classes)
                    self.outputs[("semantic", scale)] = semantic
                
                for k, v in reg_loss.items():
                    self.outputs[k, scale] = v

        elif not no_depth:
            depth, rgb_marched, semantic, _ = self.get_density(rays_o, rays_d, Voxel_feat_list[0], is_train, inputs)
            depth = depth.reshape(self.opt.cam_N, self.opt.render_h, self.opt.render_w).unsqueeze(1)
            depth = depth.clamp(self.opt.min_depth_test, self.opt.max_depth_test)
            self.outputs[("disp", 0)] = depth
            if semantic is not None:
                semantic = semantic.reshape(self.opt.cam_N, self.opt.render_h, self.opt.render_w, self.semantic_classes)
                self.outputs[("semantic", 0)] = semantic

        self.outputs[("density")] = Voxel_feat_list[0]
        
        if not is_train:
            if self.opt.dataset == 'nusc':
                H, W, Z = 200, 200, 16
                xyz_min = [-40, -40, -1]
                xyz_max = [40, 40, 5.4]
            else:
                raise NotImplementedError
            # generate the occupancy grid for test
            xyz = basic.gridcloud3d(1, Z, W, H, device='cuda').to(Voxel_feat_list[0])
            xyz_min = torch.tensor(xyz_min).to(xyz)
            xyz_max = torch.tensor(xyz_max).to(xyz)
            occ_size = torch.tensor([H, W, Z]).to(xyz)
            xyz = xyz / occ_size * (xyz_max - xyz_min) + xyz_min + 0.5 * self.voxel_size
            
            ret = self.grid_sampler(xyz, Voxel_feat_list[0], vis=True)
            if self.use_semantic:
                pred_occ_logits = ret[2]
            else:
                pred_occ_logits = ret[0]

            outputs["pred_occ_logits"] = pred_occ_logits.reshape(Z, W, H, -1).permute(3, 2, 1, 0).unsqueeze(0)

        return self.outputs
