#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import copy
import numpy as np
import torch
from topox.core.visualizer.lane_vis import draw_annotation_bev
import mmcv
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.models.builder import build_head,build_neck
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from topox.streaming.models.utils.memory_buffer import StreamTensorMemory
from ...utils.builder import build_bev_constructor
import torch.nn.functional as F
import os

@DETECTORS.register_module()
class TopoX(MVXTwoStageDetector):

    def __init__(self,
                 bev_constructor=None,
                 lane_head=None,
                 lclc_head=None,
                 bbox_head=None,
                 lcte_head=None,
                 bev_h=None,
                 bev_w=None,
                 roi_size=None,
                 video_test_mode=False,
                 streaming_cfg = dict(),
                 **kwargs):

        super(TopoX, self).__init__(**kwargs)

        if bev_constructor is not None:
            self.bev_constructor = build_bev_constructor(bev_constructor)

        if lane_head is not None:
            lane_head.update(train_cfg=self.train_cfg.lane)
            self.pts_bbox_head = build_head(lane_head)
        else:
            self.pts_bbox_head = None

        if lclc_head is not None:
            self.lclc_head = build_head(lclc_head)
        else:
            self.lclc_head = None

        if bbox_head is not None:
            bbox_head.update(train_cfg=self.train_cfg.bbox)
            self.bbox_head = build_head(bbox_head)
        else:
            self.bbox_head = None

        if lcte_head is not None:
            self.lcte_head = build_head(lcte_head)
        else:
            self.lcte_head = None

        self.fp16_enabled = False

        # BEV
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.roi_size = roi_size

        if streaming_cfg:
            self.streaming_bev = streaming_cfg['streaming_bev']
            self.stream = streaming_cfg['stream']
        else:
            self.streaming_bev = False
            self.stream = False
        if self.streaming_bev:
            self.stream_fusion_neck = build_neck(streaming_cfg['fusion_cfg'])

            self.batch_size = streaming_cfg['batch_size']
            self.bev_memory = StreamTensorMemory(
                self.batch_size,
            )

            xmin, xmax = -roi_size[0]/2, roi_size[0]/2
            ymin, ymax = -roi_size[1]/2, roi_size[1]/2
            zmin, zmax = -roi_size[2]/2, roi_size[2]/2
            x = torch.linspace(xmin, xmax, bev_w)
            y = torch.linspace(ymax, ymin, bev_h)

            y, x = torch.meshgrid(y, x)
            # z, y, x = torch.meshgrid(z, y, x)
            z = torch.zeros_like(x)
            ones = torch.ones_like(x)
            plane = torch.stack([x, y, z, ones], dim=-1)

            self.register_buffer('plane', plane.double())
            self.stream_fusion_neck.init_weights()

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def update_bev_feature(self, curr_bev_feats, img_metas):
        '''
        Args:
            curr_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
            img_metas: current image metas (List of #bs samples)
            bev_memory: where to load and store (training and testing use different buffer)
            pose_memory: where to load and store (training and testing use different buffer)

        Out:
            fused_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
        '''

        bs = curr_bev_feats.size(0)
        fused_feats_list = []

        memory = self.bev_memory.get(img_metas)
        bev_memory, pose_memory = memory['tensor'], memory['img_metas']
        is_first_frame_list = memory['is_first_frame']

        for i in range(bs):
            is_first_frame = is_first_frame_list[i]

            if self.stream:
                if is_first_frame:

                    new_feat = self.stream_fusion_neck(curr_bev_feats[i].clone().detach(), curr_bev_feats[i])
                    fused_feats_list.append(new_feat)
                else:
                    # else, warp buffered bev feature to current pose

                    prev_e2g_trans = self.plane.new_tensor(pose_memory[i]['lidar2global_translation'], dtype=torch.float64)
                    prev_e2g_rot = self.plane.new_tensor(pose_memory[i]['lidar2global_rotation'], dtype=torch.float64)
                    curr_e2g_trans = self.plane.new_tensor(img_metas[i]['lidar2global_translation'], dtype=torch.float64)
                    curr_e2g_rot = self.plane.new_tensor(img_metas[i]['lidar2global_rotation'], dtype=torch.float64)

                    prev_g2e_matrix = torch.eye(4, dtype=torch.float64, device=prev_e2g_trans.device)
                    prev_g2e_matrix[:3, :3] = prev_e2g_rot.T
                    prev_g2e_matrix[:3, 3] = -(prev_e2g_rot.T @ prev_e2g_trans)

                    curr_e2g_matrix = torch.eye(4, dtype=torch.float64, device=prev_e2g_trans.device)
                    curr_e2g_matrix[:3, :3] = curr_e2g_rot
                    curr_e2g_matrix[:3, 3] = curr_e2g_trans

                    curr2prev_matrix = prev_g2e_matrix @ curr_e2g_matrix
                    prev_coord = torch.einsum('lk,ijk->ijl', curr2prev_matrix, self.plane).float()[..., :2]

                    prev_coord[..., 0] = prev_coord[..., 0] / (self.roi_size[0]/2)
                    prev_coord[..., 1] = prev_coord[..., 1] / (self.roi_size[1]/2)

                    warped_feat = F.grid_sample(bev_memory[i].unsqueeze(0),
                                    prev_coord.unsqueeze(0),
                                    padding_mode='zeros', align_corners=False).squeeze(0)
                    new_feat = self.stream_fusion_neck(warped_feat, curr_bev_feats[i])
                    fused_feats_list.append(new_feat)
            else:

                new_feat = self.stream_fusion_neck(curr_bev_feats[i].clone().detach(), curr_bev_feats[i])
                fused_feats_list.append(new_feat)

        fused_feats = torch.stack(fused_feats_list, dim=0)

        self.bev_memory.update(fused_feats, img_metas)

        return fused_feats, is_first_frame_list

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:

            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.bev_constructor(img_feats, img_metas, prev_bev)
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_lanes_3d=None,
                      gt_lane_labels_3d=None,
                      gt_lane_adj=None,
                      gt_lane_left_type=None,
                      gt_lane_right_type=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_instance_masks=None,
                      gt_bboxes_ignore=None,
                      ):

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        if self.video_test_mode:
            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        else:
            prev_bev = None

        img_metas = [each[len_queue-1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bev_feats = self.bev_constructor(img_feats, img_metas, prev_bev)
        bev_feats = bev_feats.view(bev_feats.shape[0], self.bev_h, self.bev_w, 256)
        bev_feats = bev_feats.permute(0, 3, 1, 2)
         #torch.Size([4, 256, 100, 200])

        if self.streaming_bev:
            self.bev_memory.train()
            bev_feats,is_first_frame_list = self.update_bev_feature(bev_feats, img_metas)

        bev_feats = bev_feats.permute(0, 2, 3, 1)

        bev_feats = bev_feats.flatten(start_dim=1, end_dim=2)

        losses = dict()

        outs, dn_outs, lane_losses, lane_dn_losses, lane_assign_result = self.pts_bbox_head(img_feats, bev_feats, img_metas, gt_lanes_3d, gt_lane_labels_3d, gt_instance_masks, gt_lane_left_type, gt_lane_right_type, return_loss=True)

        #     outs, img_metas, rescale=False)

        #             center_lanes.append(center_lane)
        #             ls_mask.append(True)

        #             ls_mask.append(False)

        #         pred_area_index.append(pred_idx)

        #     os.makedirs(out_dir)

        # mmcv.imwrite(conn_img, output_path)

        for loss in lane_losses:
            losses['lane_head.' + loss] = lane_losses[loss]

        for loss in lane_dn_losses:
            losses['lane_head.' + loss] = lane_dn_losses[loss]

        lane_feats = outs['history_states']

        if self.lclc_head is not None:

            rel_logits = self.lclc_head.forward(lane_feats, lane_feats)
            lclc_losses = self.lclc_head.loss(
                rel_logits, gt_lane_adj, lane_assign_result, lane_assign_result)
            for loss in lclc_losses:
                losses['lclc_head.' + loss] = lclc_losses[loss]
            if self.pts_bbox_head.topology_memory_enabled:
                self.pts_bbox_head.store_topology_memory(
                    rel_logits.squeeze(-1).sigmoid(), img_metas)
            if self.pts_bbox_head.stream_topology_loss_enabled:
                losses['lclc_head.loss_stream_topology'] = (
                    self.lclc_head.forward_stream_train(
                        outs['stream_graph_queries'],
                        outs['stream_graph_gt_indices'], gt_lane_adj))

            lane_feats = dn_outs['history_states']
            dn_meta = dn_outs['dn_meta']

            dn_meta['gt_lane_labels_3d'] = gt_lane_labels_3d

            lclc_dn_losses = self.lclc_head.forward_dn_train(lane_feats, lane_feats, gt_lane_adj, dn_meta)
            for loss in lclc_dn_losses:
                losses['lclc_head.' + loss] = lclc_dn_losses[loss]

        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0]['can_bus'][-1] = 0
            img_metas[0]['can_bus'][:3] = 0

        new_prev_bev, results_list = self.simple_test(
            img_metas, img, prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return results_list

    def simple_test_pts(self, x, img_metas, img=None, prev_bev=None,  seq_info=None, rescale=False):
        """Test function"""
        batchsize = len(img_metas)
        scene_name, local_idx, seq_length  = seq_info[0]
        img_metas[0]['local_idx'] = local_idx
        first_frame = (local_idx == 0)

        bev_feats = self.bev_constructor(x, img_metas, prev_bev)

        bev_feats = bev_feats.view(bev_feats.shape[0], self.bev_h, self.bev_w, 256)  #1 100 200 256
        bev_feats = bev_feats.permute(0, 3, 1, 2)
         #torch.Size([4, 256, 100, 200])

        if self.streaming_bev:
            self.bev_memory.eval()
            bev_feats, is_first_frame_list = self.update_bev_feature(bev_feats, img_metas)

        bev_feats = bev_feats.permute(0, 2, 3, 1)

        bev_feats = bev_feats.flatten(start_dim=1, end_dim=2)

        outs = self.pts_bbox_head(x, bev_feats, img_metas, return_loss=False)

        lane_results = self.pts_bbox_head.get_lanes(
            outs, img_metas, rescale=rescale)

        thr_det = 0.3 if first_frame else 0.8  #0.3 0.5

        pos_results = self.pts_bbox_head.prepare_temporal_propagation(outs, scene_name, local_idx,
                                         thr_track=0.3, thr_det=thr_det) #0.4
        meta_results = img_metas

        #             center_lanes.append(center_lane)
        #             ls_mask.append(True)

        #             ls_mask.append(False)

        #         pred_area_index.append(pred_idx)

        #     os.makedirs(out_dir)

        # mmcv.imwrite(conn_img, output_path)

        if self.lclc_head is not None:
            lane_feats = outs['history_states']
            lsls_results = self.lclc_head.get_relationship(lane_feats, lane_feats)
            if self.pts_bbox_head.topology_memory_enabled:
                self.pts_bbox_head.store_topology_memory(
                    torch.stack(lsls_results), img_metas)
            lsls_results = [result.detach().cpu().numpy() for result in lsls_results]
        else:
            lsls_results = [None for _ in range(batchsize)]

        return bev_feats, lane_results, lsls_results, pos_results, meta_results

    def simple_test(self, img_metas, img=None, prev_bev=None, seq_info=None, rescale=False):
        """Test function without augmentaiton."""

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        results_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, lane_results, lsls_results, pos_results, meta_results = self.simple_test_pts(
            img_feats, img_metas, img, prev_bev, rescale=rescale, seq_info = seq_info)

        for result_dict, lane, lsls, pos, meta in zip(results_list, lane_results, lsls_results, pos_results, meta_results):

            result_dict['lane_results'] = lane
            result_dict['bbox_results'] = None
            result_dict['lsls_results'] = lsls
            result_dict['lste_results'] = None
            result_dict['pos_results'] = pos
            result_dict['meta_results'] = meta
            result_dict['sample_idx'] = img_metas[0]['sample_idx']

        return new_prev_bev, results_list

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if self.streaming_bev:
            self.bev_memory.train(*args, **kwargs)

    def eval(self):
        super().eval()
        if self.streaming_bev:
            self.bev_memory.eval()
