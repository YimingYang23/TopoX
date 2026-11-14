#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import copy
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.cnn import Linear, build_activation_layer
from mmcv.runner import auto_fp16, force_fp32
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmcv.runner import auto_fp16
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models.dense_heads import AnchorFreeHead
import matplotlib.pyplot as plt
from topox.streaming.models.utils.memory_buffer import StreamTensorMemory, StreamListMemory
from topox.streaming.models.utils.query_update import LaneSegMotionMLP
from matplotlib.path import Path
from topox.streaming.models.utils.utils import gen_sineembed_for_position, SinePositionalEncoding, gen_3D_sineembed_for_position
from topox.streaming.models.utils.query_denoising_new import LanednQueryGenerator
from topox.streaming.models.utils.dn_memory_buffer import DNStreamTensorMemory
from topox.core.visualizer.lane_vis import draw_annotation_bev
from mmcv.cnn import xavier_init
class DirectedTopologyAggregation(nn.Module):
    """Update ordered streaming lane queries with directed soft edges."""

    def __init__(self, embed_dims):
        super().__init__()
        self.pre_proj = nn.Linear(embed_dims, embed_dims)
        self.succ_proj = nn.Linear(embed_dims, embed_dims)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dims * 3, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, query, topology, valid):
        if topology.shape != (query.shape[0], query.shape[0]):
            raise ValueError('Topology and streaming-query order disagree')
        valid_pair = valid[:, None] & valid[None, :]
        topology = topology.to(query.dtype) * valid_pair.to(query.dtype)
        topology = topology.clone()
        topology.fill_diagonal_(0)
        succ = topology @ self.succ_proj(query)
        pred = topology.transpose(0, 1) @ self.pre_proj(query)
        succ = succ / topology.sum(-1, keepdim=True).clamp_min(1e-6)
        pred = pred / topology.sum(0, keepdim=True).transpose(0, 1).clamp_min(1e-6)
        delta = self.fusion(torch.cat([query, pred, succ], dim=-1))
        # Confidence gate: no incident edge => exact identity; weak edges => small update.
        incident_confidence = torch.maximum(topology.max(dim=1).values,
                                            topology.max(dim=0).values).clamp(0, 1)
        return query + 0.25 * delta * (valid.to(query.dtype) * incident_confidence)[:, None]

@HEADS.register_module()
class TopoXHead(AnchorFreeHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 dn_iter=0,
                 dn_cls_num=3,
                 tolerant_noise=0.2,
                 num_points = 10,
                 noise_decay_scale=[0.7, 0.7, 0.7],
                 stream_dn=False,
                 chamfer_thresh=None,
                 roi_size=(60, 30),
                 num_query=200,
                 with_box_refine=False,
                 with_shared_param=None,
                 transformer=None,
                 bbox_coder=None,
                 num_reg_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 pc_range=None,
                 pts_dim=3,
                 sync_cls_avg_factor=False,
                 num_lane_type_classes=3,
                 streaming_cfg=dict(),
                 dn_cfg=dict(),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_mask=dict(type='CrossEntropyLoss', loss_weight=3.0),
                 loss_dice=dict(type='DiceLoss', loss_weight=3.0),
                 loss_lane_type=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                loss_dn_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.5),
                loss_dn_bbox=dict(type='L1Loss', loss_weight=0.025),
                loss_dn_lane_type=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=0.1),
                loss_dn_mask=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    reduction='mean',
                    loss_weight=3.0),
                loss_dn_dice=dict(
                    type='DiceLoss',
                    use_sigmoid=True,
                    activate=True,
                    reduction='mean',
                    naive_dice=True,
                    eps=1.0,
                    loss_weight=3.0),
                loss_reg=dict(
                    type='LinesL1Loss',
                    loss_weight=50.0,
                    beta=0.01,
                ),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)
                     )),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 pred_mask=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'

            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_lane_type = build_loss(loss_lane_type)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)
        self.loss_dn_cls = build_loss(loss_dn_cls)
        self.loss_dn_bbox = build_loss(loss_dn_bbox)
        self.loss_dn_lane_type = build_loss(loss_dn_lane_type)
        self.loss_dn_mask = build_loss(loss_dn_mask)
        self.loss_dn_dice = build_loss(loss_dn_dice)

        self.loss_reg = build_loss(loss_reg)
        self.pred_mask = pred_mask
        self.loss_mask_type = loss_mask['type']

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if loss_lane_type.use_sigmoid:
            self.cls_lane_type_out_channels = num_lane_type_classes
        else:
            self.cls_lane_type_out_channels = num_lane_type_classes + 1

        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        assert pts_dim in (2, 3)
        self.pts_dim = pts_dim

        self.with_box_refine = with_box_refine
        if with_shared_param is not None:
            self.with_shared_param = with_shared_param
        else:
            self.with_shared_param = not self.with_box_refine
        self.as_two_stage = False

        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = pts_dim * 30
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, ] * self.code_size
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.gt_c_save = self.code_size

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_reg_fcs = num_reg_fcs
        self.num_lane_type_classes = num_lane_type_classes

        ### stream
        self.iter = 0
        self.dn_iter = dn_iter
        self.stream_dn = stream_dn
        self.tolerant_noise = tolerant_noise
        self.noise_decay_scale = noise_decay_scale
        self.chamfer_thresh = chamfer_thresh
        self.dn_cfg = dn_cfg
        self.num_queries = num_query
        self.dn_cls_num = dn_cls_num
        self.num_points = num_points
        if streaming_cfg:
            self.streaming_query = streaming_cfg['streaming']

        else:
            self.streaming_query = False

        self.topology_memory_enabled = False
        self.topology_aggregation_enabled = False
        self.stream_topology_loss_enabled = False
        if self.streaming_query:
            self.batch_size = streaming_cfg['batch_size']
            self.topk_query = streaming_cfg['topk']
            self.trans_loss_weight = streaming_cfg.get('trans_loss_weight', 0.0)
            self.topology_memory_enabled = streaming_cfg.get('topology_memory', False)
            self.topology_aggregation_enabled = streaming_cfg.get('topology_aggregation', False)
            self.stream_topology_loss_enabled = streaming_cfg.get('stream_topology_loss', False)
            if self.topology_aggregation_enabled and not self.topology_memory_enabled:
                raise ValueError('Topology aggregation requires topology memory')
            self.query_memory = StreamTensorMemory(
                self.batch_size,
            )
            self.reference_points_memory = StreamTensorMemory(
                self.batch_size,
            )
            self.lane_ref_points_memory = StreamTensorMemory(
                self.batch_size,
            )
            self.lane_id_memory = StreamListMemory(
                self.batch_size,
            )
            if self.topology_memory_enabled:
                self.topology_memory = StreamTensorMemory(self.batch_size)
                self.topology_valid_memory = StreamTensorMemory(self.batch_size)
            if self.topology_aggregation_enabled:
                self.topology_aggregation = DirectedTopologyAggregation(self.embed_dims)
            if self.stream_topology_loss_enabled:
                self.topology_lane_id_memory = StreamListMemory(self.batch_size)

            c_dim = 12

            self.query_update = LaneSegMotionMLP(c_dim=c_dim, f_dim=self.embed_dims, identity=True)
            self.target_memory = StreamTensorMemory(self.batch_size)
            self.target_label_memory = StreamTensorMemory(self.batch_size)
            self.target_left_type_memory = StreamTensorMemory(self.batch_size)
            self.target_right_type_memory = StreamTensorMemory(self.batch_size)
            self.target_mask_memory = StreamTensorMemory(self.batch_size)

        self.register_buffer('roi_size', torch.tensor(roi_size, dtype=torch.float32))
        origin = (-roi_size[0]/2, -roi_size[1]/2, self.pc_range[2])
        self.register_buffer('origin', torch.tensor(origin, dtype=torch.float32))
        self.map_size = [-50, -25, 50, 25]
        ##init
        self._init_layers()

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size // 3))
        reg_branch = nn.Sequential(*reg_branch)

        reg_branch_offset = []
        for _ in range(self.num_reg_fcs):
            reg_branch_offset.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch_offset.append(nn.ReLU())
        reg_branch_offset.append(Linear(self.embed_dims, self.code_size // 3))
        reg_branch_offset = nn.Sequential(*reg_branch_offset)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

        cls_left_type_branch = []
        for _ in range(self.num_reg_fcs):
            cls_left_type_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_left_type_branch.append(nn.LayerNorm(self.embed_dims))
            cls_left_type_branch.append(nn.ReLU(inplace=True))
        cls_left_type_branch.append(Linear(self.embed_dims, self.cls_lane_type_out_channels))
        fc_cls_left_type = nn.Sequential(*cls_left_type_branch)

        cls_right_type_branch = []
        for _ in range(self.num_reg_fcs):
            cls_right_type_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_right_type_branch.append(nn.LayerNorm(self.embed_dims))
            cls_right_type_branch.append(nn.ReLU(inplace=True))
        cls_right_type_branch.append(Linear(self.embed_dims, self.cls_lane_type_out_channels))
        fc_cls_right_type = nn.Sequential(*cls_right_type_branch)

        mask_branch = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims))

        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if not self.with_shared_param:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.reg_branches_offset = _get_clones(reg_branch_offset, num_pred)
            self.cls_left_type_branches = _get_clones(fc_cls_left_type, num_pred)
            self.cls_right_type_branches = _get_clones(fc_cls_right_type, num_pred)
            self.mask_embed = _get_clones(mask_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.reg_branches_offset = nn.ModuleList(
                [reg_branch_offset for _ in range(num_pred)])
            self.cls_left_type_branches = nn.ModuleList(
                [fc_cls_left_type for _ in range(num_pred)])
            self.cls_right_type_branches = nn.ModuleList(
                [fc_cls_right_type for _ in range(num_pred)])
            self.mask_embed = nn.ModuleList(
                [mask_branch for _ in range(num_pred)])
        ### init stream
        if self.dn_cfg is not None:

            self.dn_pts = Linear(self.embed_dims*3//4, self.embed_dims//2) ##192->128
            self.label_embedding = nn.Embedding(self.dn_cls_num, self.embed_dims//2)
            self.left_type_label_embedding = nn.Embedding(self.dn_cls_num+1, self.embed_dims//2)
            self.right_type_label_embedding = nn.Embedding(self.dn_cls_num+1, self.embed_dims//2)
            self.query_convert = Linear(self.embed_dims, self.embed_dims)
            self.instance_convert = Linear(self.embed_dims*self.num_points//2, self.embed_dims//2)
            self.content_fusion = Linear(self.embed_dims//2*3, self.embed_dims//2)
            xavier_init(self.dn_pts, distribution='uniform', bias=0.)
            xavier_init(self.query_convert, distribution='uniform', bias=0.)
            xavier_init(self.instance_convert, distribution='uniform', bias=0.)
            xavier_init(self.content_fusion, distribution='uniform', bias=0.)

        self.reference_points_embed = nn.Linear(self.embed_dims, self.pts_dim) ###
        xavier_init(self.reference_points_embed, distribution='uniform', bias=0.)

        if self.streaming_query:
            if isinstance(self.query_update, LaneSegMotionMLP):
                self.query_update.init_weights()
            if hasattr(self, 'query_alpha'):
                for m in self.query_alpha:
                    for param in m.parameters():
                        if param.dim() > 1:
                            nn.init.zeros_(param)

    def _forward_mask_head(self, output, bev_feats, lvl):
        # shape (bs, num_query, embed_dims)
        bev_feats = bev_feats.view([bev_feats.shape[0], self.bev_h, self.bev_w, self.embed_dims])
        bev_feats = bev_feats.permute(0, 3, 1, 2).contiguous()
        mask_embed = self.mask_embed[lvl](output) #torch.Size([2, 200, 256])
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, bev_feats) #torch.Size([2, 200, 100, 200])

        return outputs_mask

    def propagate(self, query_embedding, bev_feats, img_metas, same_id_gt_lane = None, return_loss=True):
        bs = query_embedding.shape[0]
        propagated_query_list = []
        stream_gt_indices_list = []
        prop_reference_points_list = []
        prop_centerline_reference_points_list= []
        prev2curr_matrix_list = []

        tmp = self.query_memory.get(img_metas)
        query_memory, pose_memory = tmp['tensor'], tmp['img_metas']
        if self.topology_memory_enabled:
            topology_memory = self.topology_memory.get(img_metas)['tensor']
            topology_valid_memory = self.topology_valid_memory.get(img_metas)['tensor']
        if return_loss and self.stream_topology_loss_enabled:
            previous_lane_ids = self.topology_lane_id_memory.get(img_metas)['tensor']

        tmp = self.reference_points_memory.get(img_metas)
        ref_pts_memory, pose_memory = tmp['tensor'], tmp['img_metas']

        tmp = self.lane_ref_points_memory.get(img_metas)
        lane_ref_pts_memory, pose_memory = tmp['tensor'], tmp['img_metas']

        if return_loss:
            target_memory = self.target_memory.get(img_metas)['tensor']
            target_label_memory = self.target_label_memory.get(img_metas)['tensor']
            target_left_type_memory = self.target_left_type_memory.get(img_metas)['tensor']
            target_right_type_memory = self.target_right_type_memory.get(img_metas)['tensor']
            trans_loss = query_embedding.new_zeros((1,))
            trans_cls_loss = query_embedding.new_zeros((1,))
            trans_left_type = query_embedding.new_zeros((1,))
            trans_right_type = query_embedding.new_zeros((1,))
            trans_loss_dice = query_embedding.new_zeros((1,))
            trans_loss_mask = query_embedding.new_zeros((1,))
            num_pos = 0

        is_first_frame_list = tmp['is_first_frame']

        for i in range(bs):
            is_first_frame = is_first_frame_list[i]
            if is_first_frame:                         ###33            256
                if return_loss and self.stream_topology_loss_enabled:
                    stream_gt_indices_list.append(torch.full(
                        (self.topk_query,), -1, dtype=torch.long,
                        device=query_embedding.device))
                padding = query_embedding.new_zeros((self.topk_query, self.embed_dims))
                propagated_query_list.append(padding)

                padding = query_embedding.new_zeros((self.topk_query, 8, 3))
                prop_reference_points_list.append(padding)

                padding = query_embedding.new_zeros((self.topk_query, 10, 3))
                prop_centerline_reference_points_list.append(padding)

                prev2curr_matrix_list.append(torch.eye(4, dtype=torch.float64).to(query_embedding.device))
            else:
                # use float64 to do precise coord transformation lidar2global_translation
                prev_e2g_trans = self.roi_size.new_tensor(pose_memory[i]['lidar2global_translation'], dtype=torch.float64)
                prev_e2g_rot = self.roi_size.new_tensor(pose_memory[i]['lidar2global_rotation'], dtype=torch.float64)
                curr_e2g_trans = self.roi_size.new_tensor(img_metas[i]['lidar2global_translation'], dtype=torch.float64)
                curr_e2g_rot = self.roi_size.new_tensor(img_metas[i]['lidar2global_rotation'], dtype=torch.float64)

                prev_e2g_matrix = torch.eye(4, dtype=torch.float64).to(query_embedding.device)
                prev_e2g_matrix[:3, :3] = prev_e2g_rot
                prev_e2g_matrix[:3, 3] = prev_e2g_trans

                curr_g2e_matrix = torch.eye(4, dtype=torch.float64).to(query_embedding.device)
                curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
                curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

                prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
                pos_encoding = prev2curr_matrix.float()[:3].view(-1)

                prev2curr_matrix_list.append(prev2curr_matrix)

                prop_q = query_memory[i]

                query_memory_updated = self.query_update(
                    prop_q, # (topk, embed_dims)
                    pos_encoding.view(1, -1).repeat(len(query_memory[i]), 1)
                ) ##topk, 256
                if self.topology_aggregation_enabled:
                    topology = topology_memory[i]
                    valid = topology_valid_memory[i]
                    if topology is not None and valid is not None:
                        query_memory_updated = self.topology_aggregation(
                            query_memory_updated, topology, valid.bool())
                if return_loss and self.stream_topology_loss_enabled:
                    previous_ids = previous_lane_ids[i]
                    current_id_to_gt = {
                        lane_id: index
                        for index, lane_id in enumerate(img_metas[i]['lane_id'])
                        if lane_id is not None
                    }
                    indices = [current_id_to_gt.get(lane_id, -1)
                               if lane_id is not None else -1
                               for lane_id in previous_ids]
                    if len(indices) != self.topk_query:
                        raise ValueError('Stored lane IDs and Top-K queries disagree')
                    stream_gt_indices_list.append(torch.tensor(
                        indices, dtype=torch.long, device=query_embedding.device))
                propagated_query_list.append(query_memory_updated.clone())

                pred = self.reg_branches[-1](query_memory_updated)# (66, 30)
                pred_offset = self.reg_branches_offset[-1](query_memory_updated)
                pred_offset = pred_offset.view(self.topk_query, self.num_points, 3)
                assert list(pred.shape) == [self.topk_query, 3*self.num_points]

                # ref pts
                prev_ref_pts = ref_pts_memory[i][-1,:,:,:]
                assert list(prev_ref_pts.shape) == [self.topk_query, self.num_points, 3]

                denormed_last_layer_prev_ref_pts = prev_ref_pts * self.roi_size + self.origin

                denormed_last_layer_prev_ref_pts = torch.cat([
                    denormed_last_layer_prev_ref_pts,
                    denormed_last_layer_prev_ref_pts.new_ones((self.topk_query, 10, 1)) # 4-th dim
                ], dim=-1) # (num_prop, num_pts, 4)
                normed_last_layer_prev2curr_ref_pts = torch.einsum('lk,ijk->ijl', prev2curr_matrix, denormed_last_layer_prev_ref_pts.double()).float()

                normed_last_layer_prev2curr_ref_pts = (normed_last_layer_prev2curr_ref_pts[..., :3] - self.origin) / self.roi_size # (num_prop, num_pts, 2)
                normed_last_layer_prev2curr_ref_pts = torch.clip(normed_last_layer_prev2curr_ref_pts, min=0., max=1.)
                prop_centerline_reference_points_list.append(normed_last_layer_prev2curr_ref_pts)

                pred = pred.view(self.topk_query, self.num_points, 3)

                pred = pred + inverse_sigmoid(normed_last_layer_prev2curr_ref_pts)
                pred = pred.sigmoid().clone()
                denormed_pred = pred * self.roi_size + self.origin
                denormed_pred = denormed_pred.contiguous()
                denormed_pred_left = denormed_pred + pred_offset
                denormed_pred_right = denormed_pred - pred_offset

                normed_pred = torch.cat([denormed_pred, denormed_pred_left, denormed_pred_right], dim=1)

                assert list(normed_pred.shape) == [self.topk_query, self.num_points*3, 3]

                denormed_lane_ref_pts = lane_ref_pts_memory[i][-1,:,:,:] * self.roi_size + self.origin # (num_prop, num_pts, 2)

                assert list(denormed_lane_ref_pts.shape) == [self.topk_query, 8, 3]
                denormed_lane_ref_pts = torch.cat([
                    denormed_lane_ref_pts,
                    denormed_lane_ref_pts.new_ones((self.topk_query, 8, 1)) # 4-th dim
                ], dim=-1) # (num_prop, num_pts, 4)
                assert list(denormed_lane_ref_pts.shape) == [self.topk_query, 8, 4]

                curr_lane_ref_pts = torch.einsum('lk,ijk->ijl', prev2curr_matrix, denormed_lane_ref_pts.double()).float()
                normed_lane_ref_pts = (curr_lane_ref_pts[..., :3] - self.origin) / self.roi_size # (num_prop, num_pts, 2)

                normed_lane_ref_pts = torch.clip(normed_lane_ref_pts, min=0., max=1.)

                prop_reference_points_list.append(normed_lane_ref_pts)

                if return_loss:

                    targets = target_memory[i]
                                                ###topk 30 3
                    target_labels = target_label_memory[i]
                    target_left_types = target_left_type_memory[i]
                    target_right_types = target_right_type_memory[i]

                    same_id_gt_lane_single = same_id_gt_lane[i]

                    weights = targets.new_ones((self.topk_query, 9*self.num_points)) #66, 30
                    label_weights = torch.ones_like(target_labels).to(torch.float32)
                    bg_idx = torch.all(targets.view(self.topk_query, -1) == 0.0, dim=1)
                    num_pos = (self.topk_query - bg_idx.sum())
                    weights[bg_idx, :] = 0.0
                    pos_idx = torch.nonzero(~bg_idx, as_tuple=True)[0]
                    label_weights[bg_idx] = 0.0

                    denormed_targets = targets
                    denormed_targets = torch.cat([
                        denormed_targets,
                        denormed_targets.new_ones((self.topk_query, 3* self.num_points, 1)) # 4-th dim
                    ], dim=-1) # (num_prop, num_pts, 4)
                    assert list(denormed_targets.shape) == [self.topk_query, 3* self.num_points, 4]
                    curr_targets = torch.einsum('lk,ijk->ijl', prev2curr_matrix.float(), denormed_targets)

                    normed_targets = curr_targets[..., :3]
                    # (num_prop, 2*num_pts)

                    if isinstance(same_id_gt_lane_single, torch.Tensor):
                        for idx, pos_id in enumerate(pos_idx):
                            if torch.all(same_id_gt_lane_single[idx] == 0.0):
                                continue
                            else:
                                normed_targets[pos_id] = same_id_gt_lane_single.reshape(len(same_id_gt_lane_single),30,3)[idx]

                    elif same_id_gt_lane_single == [None]:
                        pass
                    else:
                        pass  # Keep inference silent.
                        raise RuntimeError(f"unexpected value in same_id_gt_lane_single")

                    trans_loss += self.loss_reg(normed_pred.reshape(-1,9*self.num_points), normed_targets.reshape(-1,9*self.num_points), weights, avg_factor=num_pos)

                    output_class = self.cls_branches[-1](query_memory_updated)
                    output_left_type = self.cls_left_type_branches[-1](query_memory_updated)
                    output_right_type = self.cls_right_type_branches[-1](query_memory_updated)

                    trans_cls_loss += self.loss_cls(output_class, target_labels, label_weights, avg_factor=num_pos)
                    trans_left_type += self.loss_lane_type(output_left_type, target_left_types, label_weights, avg_factor=num_pos)
                    trans_right_type += self.loss_lane_type(output_right_type, target_right_types, label_weights, avg_factor=num_pos)

                    pos_target_lanes = normed_targets[~bg_idx].cpu().numpy()

                    if len(pos_target_lanes)!=0:
                        trans_mask = self.generate_lanesegment_instance_mask(pos_target_lanes,device = output_class.device)
                        outputs_mask = self._forward_mask_head(query_memory_updated[~bg_idx].unsqueeze(0), bev_feats[i].unsqueeze(0), -1)
                        outputs_mask = outputs_mask.squeeze(0)
                        trans_loss_dice += self.loss_dice(
                        outputs_mask, trans_mask, avg_factor=num_pos
                        )
                        outputs_mask = outputs_mask.reshape(-1, 1)
                        trans_mask = trans_mask.reshape(-1)

                        h, w = outputs_mask.shape[-2:]

                        if self.loss_mask_type == 'FocalLoss':
                            trans_mask = (1 - trans_mask).long()
                        if self.loss_mask_type == 'CrossEntropyLoss':
                            trans_mask = trans_mask.reshape(outputs_mask.shape).bool()

                        trans_loss_mask += self.loss_mask(
                            outputs_mask, trans_mask, avg_factor=num_pos * h * w
                        )

        prop_query_embedding = torch.stack(propagated_query_list) # (bs, topk, embed_dims)
        prop_ref_pts = torch.stack(prop_reference_points_list)
        prop_centerline_ref_pts = torch.stack(prop_centerline_reference_points_list)
        assert list(prop_query_embedding.shape) == [bs, self.topk_query, self.embed_dims]

        assert list(prop_ref_pts.shape) == [bs, self.topk_query, 8, 3]
        assert list(prop_centerline_ref_pts.shape) == [bs, self.topk_query, 10, 3]

        query_pos, _ = torch.split(query_embedding, self.embed_dims, dim=2)
        init_reference_points = self.reference_points_embed(query_pos) # (bs, num_q, 2*num_pts)

        init_reference_points = init_reference_points.repeat(1, 1, self.num_points)
        init_reference_points = init_reference_points.sigmoid()
        bs, num_qeury, _ = init_reference_points.shape
        init_reference_points = init_reference_points.view(bs, num_qeury, self.num_points, self.pts_dim)
        memory_query_embedding = None

        if return_loss:

            trans_loss = self.trans_loss_weight * (trans_loss +  trans_cls_loss + trans_left_type + trans_right_type + trans_loss_dice + trans_loss_mask)
            return query_embedding, prop_query_embedding, init_reference_points, prop_ref_pts, prop_centerline_ref_pts, memory_query_embedding, is_first_frame_list, trans_loss, prev2curr_matrix_list, stream_gt_indices_list
        else:
            return query_embedding, prop_query_embedding, init_reference_points, prop_ref_pts, prop_centerline_ref_pts, memory_query_embedding, is_first_frame_list

    def denoise(self, gt_lanes_3d, gt_lane_labels_3d, gt_lane_left_type, gt_lane_right_type, gt_instance_masks, origin, roi_size, device):

        gt_pts_list = [gt_bboxes.reshape(-1, 3*self.num_points, 3) for gt_bboxes in gt_lanes_3d]

        gt_bboxes_list = []
        for gt in gt_pts_list:
            gt_bboxes_list.append(torch.cat((gt.min(1)[0], gt.max(1)[0]), -1))

        if self.dn_cfg is not None:
            self.dn_generator = LanednQueryGenerator(**self.dn_cfg)

            dn_query_line_type_content, dn_query_left_type_content, dn_query_right_type_content, input_query_bbox, input_query_mask, attn_mask, dn_meta, denoise_centerline_refers, denoise_centerline_refers_pe, denoise_lane_refers \
                            = self.dn_generator(gt_bboxes_list, gt_pts_list, gt_lane_labels_3d, gt_lane_left_type,
                                gt_lane_right_type,  self.label_embedding, self.left_type_label_embedding, self.right_type_label_embedding, origin = origin, roi_size = roi_size)

            dn_query_content = self.content_fusion(torch.cat((dn_query_line_type_content, dn_query_left_type_content, dn_query_right_type_content), -1)) #torch.Size([2, 41, 768]) -> torch.Size([2, 41, 256])

            pe_pts = self.dn_pts(gen_3D_sineembed_for_position(denoise_centerline_refers_pe.flatten(1, 2), self.embed_dims//4))

            pe_pts = pe_pts.reshape(pe_pts.size(0), -1, self.num_points, pe_pts.size(2)).flatten(2)
            dn_query_pos = self.instance_convert(pe_pts)

            query = self.query_convert(torch.cat((dn_query_content, dn_query_pos), -1))

            return query, attn_mask, dn_meta, denoise_centerline_refers, denoise_lane_refers

    def stream_denoise(self, gt_vecs_list, gt_labels_list_x, cur_gt_vecs_list, cur_gt_labels_list_x, device, prev2curr_matrix_list, noise_scale, prop_query_embedding):
        pos_valid = []
        gt_pts_list = []
        gt_labels_list = []
        last_save_idx = {}
        noise_scale_list = []

        for i, gt_bboxes in enumerate(gt_vecs_list):
            if gt_bboxes is not None:
                prev_ref_pts = gt_bboxes[:, 0, :].reshape(-1, self.num_points, 2)
                denormed_ref_pts = prev_ref_pts * self.roi_size + self.origin # (num_prop, num_pts, 2)

                denormed_ref_pts = torch.cat([
                    denormed_ref_pts,
                    denormed_ref_pts.new_zeros((denormed_ref_pts.size(0), self.num_points, 1)), # z-axis
                    denormed_ref_pts.new_ones((denormed_ref_pts.size(0), self.num_points, 1)) # 4-th dim
                ], dim=-1) # (num_prop, num_pts, 4)

                prev2curr_matrix = prev2curr_matrix_list[i]
                curr_ref_pts = torch.einsum('lk,ijk->ijl', prev2curr_matrix, denormed_ref_pts.double()).float()
                normed_ref_pts = (curr_ref_pts[..., :2] - self.origin) / self.roi_size # (num_prop, num_pts, 2)
                normed_ref_pts = torch.clip(normed_ref_pts, min=0., max=1.)
                select_idx = (((normed_ref_pts==0).sum((1, 2)) + (normed_ref_pts==1).sum((1, 2))) != 40)

                if select_idx.sum() == 0:
                    gt_pts_list.append(cur_gt_vecs_list[i][:, 0, :].reshape(-1, self.num_points, 2))
                    gt_labels_list.append(cur_gt_labels_list_x[i])
                    noise_scale_list.append(torch.ones_like(cur_gt_labels_list_x[i]).type_as(cur_gt_vecs_list[i]))
                else:
                    pos_valid.append(i)
                    gt_pts_list.append(normed_ref_pts[select_idx])
                    gt_labels_list.append(gt_labels_list_x[i][select_idx])
                    last_save_idx[i] = select_idx
                    noise_scale_list.append(noise_scale[i][select_idx])

            else:
                gt_pts_list.append(cur_gt_vecs_list[i][:, 0, :].reshape(-1, self.num_points, 2))
                gt_labels_list.append(cur_gt_labels_list_x[i])
                noise_scale_list.append(torch.ones_like(cur_gt_labels_list_x[i]).type_as(cur_gt_vecs_list[i]))

        gt_bboxes_list = []
        for gt in gt_pts_list:
            gt_bboxes_list.append(torch.cat((gt.min(1)[0], gt.max(1)[0]), -1))

        if self.dn_cfg is not None:
            self.dn_generator = LanednQueryGenerator(**self.dn_cfg)

            dn_query_content, input_query_bbox, input_query_pts, attn_mask, dn_meta, denoise_refers \
                            = self.dn_generator(gt_bboxes_list, gt_pts_list, gt_labels_list, self.label_embedding, prop_query_embedding, noise_scale_list)
            pe_pts = self.dn_pts(gen_sineembed_for_position(denoise_refers.flatten(1, 2), self.embed_dims//4))
            pe_pts = pe_pts.reshape(pe_pts.size(0), -1, self.num_points, pe_pts.size(2)).flatten(2)
            dn_query_pos = self.instance_convert(pe_pts)
            query = self.query_convert(torch.cat((dn_query_content, dn_query_pos), -1))

            dn_meta['stream_valid'] = pos_valid
            dn_meta['last_valid_idx'] = last_save_idx
            return query, attn_mask, dn_meta, denoise_refers

    def generate_lanesegment_instance_mask(self, results, device):
        gt_lanes = results.reshape(-1, 3, 10, 3)
        gt_left_lines = gt_lanes[:, 1]
        gt_right_lines = gt_lanes[:, 2]

        origin = np.array([self.bev_w // 2, self.bev_h // 2])
        scale = np.array([self.bev_w / (self.map_size[2] - self.map_size[0]), self.bev_h / (self.map_size[3] - self.map_size[1])])

        inst_masks = []
        for idx, (left_line, right_line) in enumerate(zip(gt_left_lines, gt_right_lines)):

            segment_boundary = np.concatenate((left_line, right_line[::-1], left_line[0:1]), axis=0)
            mask = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)

            draw_coor = (segment_boundary[:, :2] * scale + origin).astype(np.int32)
            mask = cv2.fillPoly(mask, [draw_coor], 255)
            bitMask = (mask / 255)
            bitMask = torch.tensor(bitMask).to(device=device)
            inst_masks.append(bitMask)
        inst_masks = torch.stack(inst_masks, 0)

        return inst_masks

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward_train(self, mlvl_feats, bev_feats, img_metas,gt_lanes_3d = None, gt_lane_labels_3d = None, gt_instance_masks = None, gt_lane_left_type = None, gt_lane_right_type = None):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_lanes_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 99].
            all_mask_preds (Tensor): Sigmoid outputs from the segmentation \
                head with normalized value in the range of [0,1].
                Shape []
        """

        bs = len(img_metas)
        dtype = mlvl_feats[0].dtype

        object_query_embeds = self.query_embedding.weight.to(dtype)[None, ...].repeat(bs, 1, 1)
        if self.streaming_query:
            same_id_gt_lane = []
            tmp = self.lane_id_memory.get(img_metas)
            last_lane_id, pose_memory = tmp['tensor'], tmp['img_metas']
            match_current_gt_target = None
            for i in range(bs):
                if last_lane_id[i]:
                    _last_lane_id = last_lane_id[i]
                    _current_lane_id = img_metas[i]['lane_id']
                    match_current_gt_target = torch.zeros((len(_last_lane_id), 90)).to(object_query_embeds.device)
                    list2_indices = {value: idx for idx, value in enumerate(_current_lane_id)}
                    match_indices = [list2_indices.get(item, -1) for item in _last_lane_id]
                    for j in range(len(_last_lane_id)):
                        if match_indices[j]!=-1:
                            match_current_gt_target[j] = gt_lanes_3d[i][match_indices[j]]
                    same_id_gt_lane.append(match_current_gt_target)
                else:

                    same_id_gt_lane.append([None])
            query_embedding, prop_query_embedding, init_reference_points, prop_ref_pts, prop_centerline_ref_pts, memory_query, is_first_frame_list, trans_loss, prev2curr_matrix_list, stream_gt_indices_list = \
                self.propagate(object_query_embeds, bev_feats, img_metas, same_id_gt_lane, return_loss=True)

        else:
            query_pos, _ = torch.split(object_query_embeds, self.embed_dims, dim=2)

            init_reference_points = self.reference_points_embed(query_pos) # torch.Size([2, 200, 256])-> torch.Size([2, 200, 3])

            init_reference_points = init_reference_points.repeat(1, 1, self.num_points) #torch.Size([2, 200, 3])-> torch.Size([2, 200, 30])
            init_reference_points = init_reference_points.sigmoid()
            bs, num_qeury, _ = init_reference_points.shape
            init_reference_points = init_reference_points.view(bs, num_qeury, self.num_points, self.pts_dim)
            prop_query_embedding = None
            prop_ref_pts = None
            prop_centerline_ref_pts = None
            is_first_frame_list = [True for i in range(bs)]
            query_embedding = object_query_embeds

        assert list(init_reference_points.shape) == [bs, self.num_queries, self.num_points, self.pts_dim] #[1, 200, 10, 3]

        # last_gt_list, last_label_list = last_gt['gt'], last_gt['label_list']

        if True:

            #     stream_dn_query, stream_self_attn_mask, stream_dn_meta, stream_denoise_refers \
            #                 = self.stream_denoise(copy.deepcopy(last_gt_list), copy.deepcopy(last_label_list),
            #                 copy.deepcopy(gts[0]['lines']), copy.deepcopy(gts[0]['labels']),
            #                 bev_features.device, prev2curr_matrix_list, noise_scale,

            stream_dn_query, stream_self_attn_mask, stream_dn_meta, stream_centerline_denoise_refers, stream_denoise_lane_refers \
                        = self.denoise(
                        copy.deepcopy(gt_lanes_3d), copy.deepcopy(gt_lane_labels_3d),
                        copy.deepcopy(gt_lane_left_type), copy.deepcopy(gt_lane_right_type), copy.deepcopy(gt_instance_masks), origin= self.origin, roi_size = self.roi_size,
                        device = mlvl_feats[0].device)
            dn_num = stream_dn_meta['pad_size']
        else:
            stream_self_attn_mask = None
            stream_dn_query = None
            stream_denoise_refers = None
            dn_num = 0

        outputs = self.transformer(
            mlvl_feats,
            bev_feats,
            object_query_embeds,
            reference_points = init_reference_points,
            prop_query = prop_query_embedding,
            prop_reference_points = prop_ref_pts,
            prop_centerline_reference_points = prop_centerline_ref_pts,
            is_first_frame_list=is_first_frame_list,

            dn_query=stream_dn_query,
            dn_centerline_refer=stream_centerline_denoise_refers,
            dn_lane_refer=stream_denoise_lane_refers,
            attn_masks=stream_self_attn_mask,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            reg_branches=(self.reg_branches, self.reg_branches_offset) if self.with_box_refine else None,  # noqa:E501
            cls_branches=self.cls_branches,
            img_metas=img_metas
        )

        hs, init_reference, inter_references, inter_lanepts_references = outputs
        hs = hs.permute(0, 2, 1, 3)

        outputs_classes = []
        outputs_coord = []
        outputs_masks = []
        output_left_types = []
        output_right_types = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference[:, dn_num:,:]
            else:
                reference = inter_references[lvl - 1][:, dn_num:,:]
            reference = inverse_sigmoid(reference)
            assert reference.shape[-1] == self.pts_dim

            outputs_class = self.cls_branches[lvl](hs[lvl][:, dn_num:,:])
            output_left_type = self.cls_left_type_branches[lvl](hs[lvl][:, dn_num:,:])
            output_right_type = self.cls_right_type_branches[lvl](hs[lvl][:, dn_num:,:])

            tmp = self.reg_branches[lvl](hs[lvl][:, dn_num:,:])
            bs, num_query, _ = tmp.shape
            tmp = tmp.view(bs, num_query, -1, self.pts_dim)
            tmp = tmp + reference
            tmp = tmp.sigmoid()

            coord = tmp.clone()
            coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0] ##*(51.2 - -51.2) + -51.2
            coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            if self.pts_dim == 3:
                coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            centerline = coord.view(bs, num_query, -1).contiguous()

            offset = self.reg_branches_offset[lvl](hs[lvl][:, dn_num:,:])
            left_laneline = centerline + offset
            right_laneline = centerline - offset

            # segmentation head
            outputs_mask = self._forward_mask_head(hs[lvl][:, dn_num:,:], bev_feats, lvl)

            outputs_classes.append(outputs_class)
            outputs_coord.append(torch.cat([centerline, left_laneline, right_laneline], axis=-1))
            outputs_masks.append(outputs_mask)
            output_left_types.append(output_left_type)
            output_right_types.append(output_right_type)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coord)
        outputs_masks = torch.stack(outputs_masks)
        output_left_types = torch.stack(output_left_types)
        output_right_types = torch.stack(output_right_types)

        outs = {
            'all_cls_scores': outputs_classes,
            'all_lanes_preds': outputs_coords,
            'all_mask_preds': outputs_masks,
            'all_lanes_left_type': output_left_types,
            'all_lanes_right_type': output_right_types,
            'history_states': hs[:,:, dn_num:,:]
        }
        if self.streaming_query and self.stream_topology_loss_enabled:
            outs['stream_graph_queries'] = prop_query_embedding
            outs['stream_graph_gt_indices'] = stream_gt_indices_list

        loss_inputs = [outs, gt_lanes_3d, gt_lane_labels_3d, gt_instance_masks, gt_lane_left_type, gt_lane_right_type]

        lane_losses, lane_assign_result, bbox_targets, labels_targets, labels_left_type_targets, labels_right_type_targets, mask_targets, mask_targets_weights = self.loss(*loss_inputs, img_metas=img_metas)

        outputs_classes = []
        outputs_coord = []
        outputs_masks = []
        output_left_types = []
        output_right_types = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference[:, :dn_num,:]
            else:
                reference = inter_references[lvl - 1][:, :dn_num,:]
            reference = inverse_sigmoid(reference)
            assert reference.shape[-1] == self.pts_dim

            outputs_class = self.cls_branches[lvl](hs[lvl][:, :dn_num,:])
            output_left_type = self.cls_left_type_branches[lvl](hs[lvl][:, :dn_num,:])
            output_right_type = self.cls_right_type_branches[lvl](hs[lvl][:, :dn_num,:])

            tmp = self.reg_branches[lvl](hs[lvl][:, :dn_num,:])
            bs, num_query, _ = tmp.shape
            tmp = tmp.view(bs, num_query, -1, self.pts_dim)
            tmp = tmp + reference
            tmp = tmp.sigmoid()

            coord = tmp.clone()
            coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0] ##*(51.2 - -51.2) + -51.2
            coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            if self.pts_dim == 3:
                coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            centerline = coord.view(bs, num_query, -1).contiguous()

            offset = self.reg_branches_offset[lvl](hs[lvl][:, :dn_num,:])
            left_laneline = centerline + offset
            right_laneline = centerline - offset

            # segmentation head
            outputs_mask = self._forward_mask_head(hs[lvl][:, :dn_num,:], bev_feats, lvl)

            outputs_classes.append(outputs_class)
            outputs_coord.append(torch.cat([centerline, left_laneline, right_laneline], axis=-1))
            outputs_masks.append(outputs_mask)
            output_left_types.append(output_left_type)
            output_right_types.append(output_right_type)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coord)
        outputs_masks = torch.stack(outputs_masks)
        output_left_types = torch.stack(output_left_types)
        output_right_types = torch.stack(output_right_types)

        dn_outs = {
            'all_cls_scores': outputs_classes,
            'all_lanes_preds': outputs_coords,
            'all_mask_preds': outputs_masks,
            'all_lanes_left_type': output_left_types,
            'all_lanes_right_type': output_right_types,
            'history_states': hs[:,:, :dn_num,:],
            'dn_meta': stream_dn_meta
        }

        loss_inputs = [dn_outs, gt_lanes_3d, gt_lane_labels_3d, gt_instance_masks, gt_lane_left_type, gt_lane_right_type]
        lane_dn_losses = self.loss_dn(*loss_inputs, img_metas=img_metas, dn_meta=stream_dn_meta)

        if self.streaming_query:
            query_list = []
            topk_idx_list = []
            topk_valid_list = []
            topology_lane_ids_list = []
            ref_pts_list = []
            lane_ref_pts_list = []
            gt_targets_list = []
            gt_label_target_list = []
            gt_left_type_target_list = []
            gt_right_type_target_list = []
            gt_lane_id_list = []

            lines, scores = outs['all_lanes_preds'][-1], outs['all_cls_scores'][-1]
            gt_lines = bbox_targets[-1] # take results from the last layer
            gt_lines = gt_lines.reshape(bs, self.num_query, 90)
            gt_label_target = labels_targets[-1]
            gt_label_target = gt_label_target.reshape(bs, self.num_query)
            gt_left_type_target = labels_left_type_targets[-1]
            gt_left_type_target = gt_left_type_target.reshape(bs, self.num_query)
            gt_right_type_target = labels_right_type_targets[-1]
            gt_right_type_target = gt_right_type_target.reshape(bs, self.num_query)

            gt_mask_target = mask_targets[-1]
            gt_mask_weight_target = mask_targets_weights[-1]
            last_lane_assign_result = lane_assign_result[-1]

            #     gt_mask_target_new[pos_inds] = gt_mask_target_single
            #     gt_mask_target_expand.append(gt_mask_target_new)

            for i in range(bs):

                inter_queries = hs
                _lines = lines[i] #torch.Size([200, 90])
                _queries = inter_queries[-1][i][dn_num:] ## 200, 256
                _scores = scores[i] #torch.Size([200, 2])
                _class_logits = _scores
                _gt_targets = gt_lines[i] # (num_q or num_q+topk, 20, 2)
                _gt_label_target = gt_label_target[i]
                _gt_left_type_target = gt_left_type_target[i]
                _gt_right_type_target = gt_right_type_target[i]

                _init_reference = init_reference[i][dn_num:,:,:]
                _inter_reference = inter_references[:,i,dn_num:,:,:]
                _inter_lane_reference = inter_lanepts_references[:,i,dn_num:,:,:]
                _last_lane_assign_result_pos_inds= last_lane_assign_result['pos_inds'][i]
                _last_lane_assign_result_pos_assigned_gt_inds= last_lane_assign_result['pos_assigned_gt_inds'][i]
                _gt_lane_id = img_metas[i]['lane_id']
                assert len(_lines) == len(_queries)
                assert len(_lines) == len(_gt_targets)

                _scores, _ = _scores.max(-1)
                topk_score, topk_idx = _scores.topk(k=self.topk_query, dim=-1)
                topk_idx_list.append(topk_idx)
                topk_valid_list.append(_class_logits.argmax(-1)[topk_idx] == 0)
                if self.stream_topology_loss_enabled:
                    assigned_ids = {}
                    for query_idx, gt_idx in zip(
                            _last_lane_assign_result_pos_inds.tolist(),
                            _last_lane_assign_result_pos_assigned_gt_inds.tolist()):
                        if 0 <= gt_idx < len(_gt_lane_id) and gt_lane_labels_3d[i][gt_idx] == 0:
                            assigned_ids[query_idx] = _gt_lane_id[gt_idx]
                    topology_lane_ids_list.append([
                        assigned_ids.get(int(query_idx)) for query_idx in topk_idx])

                _queries = _queries[topk_idx] # (topk, embed_dims)
                _lines = _lines[topk_idx] # (topk, 2*num_pts)

                _gt_lane_id2target_id = [_gt_lane_id[i] if i < len(_gt_lane_id) else None
                for i in _last_lane_assign_result_pos_assigned_gt_inds.cpu().tolist()
                ]

                _mask_pos_in_topk = (_last_lane_assign_result_pos_inds.unsqueeze(1) == topk_idx).any(dim=1)
                _pos_in_topk = _last_lane_assign_result_pos_inds[_mask_pos_in_topk]

                indices_sort_after_topk = torch.tensor([torch.where(topk_idx == x)[0].item() for x in _pos_in_topk])

                sorted_indices = torch.argsort(indices_sort_after_topk)
                _pos_in_topk_sort = _pos_in_topk[sorted_indices]

                indices_to_keep = torch.nonzero(_mask_pos_in_topk, as_tuple=True)[0].cpu().tolist()
                _last_lane_assign_result_pos_assigned_gt_inds_topk = _last_lane_assign_result_pos_assigned_gt_inds[_mask_pos_in_topk]
                _gt_lane_id2target_id = [_gt_lane_id2target_id[i] for i in indices_to_keep]
                _gt_lane_id2target_id = [_gt_lane_id2target_id[i] for i in sorted_indices.tolist()]
                #gt_lanes_3d

                _gt_targets = _gt_targets[topk_idx] # (topk, 20, 2)

                _gt_label_target = _gt_label_target[topk_idx]
                _gt_left_type_target = _gt_left_type_target[topk_idx]
                _gt_right_type_target = _gt_right_type_target[topk_idx]

                _init_reference = _init_reference[topk_idx]
                _inter_reference = _inter_reference[:,topk_idx,:,:]#layer, topk, 10, 3
                _inter_lane_reference = _inter_lane_reference[:,topk_idx,:,:]#layer, topk, 8, 3
                _all_reference = torch.cat((_init_reference.unsqueeze(0), _inter_reference), dim=0)
                query_list.append(_queries)

                ref_pts_list.append(_all_reference)
                lane_ref_pts_list.append(_inter_lane_reference)
                gt_targets_list.append(_gt_targets.view(-1, self.num_points*3, 3))
                gt_lane_id_list.append(_gt_lane_id2target_id)
                gt_label_target_list.append(_gt_label_target)
                gt_left_type_target_list.append(_gt_left_type_target)
                gt_right_type_target_list.append(_gt_right_type_target)

            self.query_memory.update(query_list, img_metas)
            self._topk_indices = topk_idx_list
            self._topk_valid = topk_valid_list
            if self.stream_topology_loss_enabled:
                self.topology_lane_id_memory.update(topology_lane_ids_list, img_metas)
            self.reference_points_memory.update(ref_pts_list, img_metas)
            self.lane_ref_points_memory.update(lane_ref_pts_list, img_metas)
            self.target_memory.update(gt_targets_list, img_metas) #torch.Size([66, 30, 3])
            self.lane_id_memory.update(gt_lane_id_list, img_metas)
            self.target_label_memory.update(gt_label_target_list, img_metas)
            self.target_left_type_memory.update(gt_left_type_target_list, img_metas)
            self.target_right_type_memory.update(gt_right_type_target_list, img_metas)

            lane_losses['trans_loss'] = trans_loss

        self.iter += 1
        return outs, dn_outs, lane_losses, lane_dn_losses, lane_assign_result
        # return outs, trans_loss

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward_test(self, mlvl_feats, bev_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_lanes_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 99].
            all_mask_preds (Tensor): Sigmoid outputs from the segmentation \
                head with normalized value in the range of [0,1].
                Shape []
        """

        bs = len(img_metas)
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)[None, ...].repeat(bs, 1, 1)

        if self.streaming_query:
            query_embedding, prop_query_embedding, init_reference_points, prop_ref_pts, prop_centerline_ref_pts, memory_query, is_first_frame_list = \
                self.propagate(object_query_embeds, bev_feats, img_metas, return_loss=False)

        else:
            query_pos, _ = torch.split(object_query_embeds, self.embed_dims, dim=1)
            init_reference_points = self.reference_points_embed(query_pos) # (bs, num_q, num_pts)
            init_reference_points = init_reference_points.repeat(1, 1, self.num_points)
            init_reference_points = init_reference_points.sigmoid()
            bs, num_qeury, _ = init_reference_points.shape
            init_reference_points = init_reference_points.view(bs, num_qeury, self.num_points, self.pts_dim)
            prop_query_embedding = None
            prop_ref_pts = None
            prop_centerline_ref_pts = None
            is_first_frame_list = [True for i in range(bs)]

        assert list(init_reference_points.shape) == [bs, self.num_queries, self.num_points, self.pts_dim] #[1, 200, 10, 3]

        outputs = self.transformer(
            mlvl_feats,
            bev_feats,
            object_query_embeds,
            reference_points = init_reference_points,
            prop_query = prop_query_embedding,
            prop_reference_points = prop_ref_pts,
            prop_centerline_reference_points = prop_centerline_ref_pts,
            is_first_frame_list=is_first_frame_list,

            bev_h=self.bev_h,
            bev_w=self.bev_w,
            reg_branches=(self.reg_branches, self.reg_branches_offset) if self.with_box_refine else None,  # noqa:E501
            cls_branches=self.cls_branches,
            img_metas=img_metas
        )

        hs, init_reference, inter_references, inter_lanepts_references = outputs
        hs = hs.permute(0, 2, 1, 3)

        reference = inter_references[-1]
        reference = inverse_sigmoid(reference)
        assert reference.shape[-1] == self.pts_dim

        outputs_class = self.cls_branches[-1](hs[-1])
        output_left_type = self.cls_left_type_branches[-1](hs[-1])
        output_right_type = self.cls_right_type_branches[-1](hs[-1])

        tmp = self.reg_branches[-1](hs[-1])
        bs, num_query, _ = tmp.shape
        tmp = tmp.view(bs, num_query, -1, self.pts_dim)
        tmp = tmp + reference
        tmp = tmp.sigmoid()

        coord = tmp.clone()
        coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        if self.pts_dim == 3:
            coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        centerline = coord.view(bs, num_query, -1).contiguous()

        offset = self.reg_branches_offset[-1](hs[-1])
        left_laneline = centerline + offset
        right_laneline = centerline - offset

        # segmentation head
        if self.pred_mask:
            outputs_mask = self._forward_mask_head(hs[-1], bev_feats, -1)

        outputs_classes = torch.stack([outputs_class])
        outputs_coords = torch.stack([torch.cat([centerline, left_laneline, right_laneline], axis=-1)])
        output_left_types = torch.stack([output_left_type])
        output_right_types = torch.stack([output_right_type])

        outs = {
            'all_cls_scores': outputs_classes,
            'all_lanes_preds': outputs_coords,
            'all_mask_preds': torch.stack([outputs_mask]) if self.pred_mask else None,
            'all_lanes_left_type': output_left_types,
            'all_lanes_right_type': output_right_types,
            'history_states': hs,
            'sample_idx':img_metas[0]['sample_idx']
        }

        if self.streaming_query:
            query_list = []
            topk_idx_list = []
            topk_valid_list = []
            ref_pts_list = []
            lane_ref_pts_list = []

            lines, scores = outs['all_lanes_preds'][-1], outs['all_cls_scores'][-1]

            for i in range(bs):

                inter_queries = hs
                _lines = lines[i] #torch.Size([200, 90])
                _queries = inter_queries[-1][i]
                _scores = scores[i] #torch.Size([200, 2])
                _class_logits = _scores

                _init_reference = init_reference[i]
                _inter_reference = inter_references[:,i,:,:,:]
                _inter_lane_reference = inter_lanepts_references[:,i,:,:,:]
                assert len(_lines) == len(_queries)

                _scores, _ = _scores.max(-1)
                topk_score, topk_idx = _scores.topk(k=self.topk_query, dim=-1)
                topk_idx_list.append(topk_idx)
                topk_valid_list.append(_class_logits.argmax(-1)[topk_idx] == 0)

                _queries = _queries[topk_idx] # (topk, embed_dims)
                _lines = _lines[topk_idx] # (topk, 2*num_pts)

                _init_reference = _init_reference[topk_idx]
                _inter_reference = _inter_reference[:,topk_idx,:,:]#layer, topk, 10, 3
                _inter_lane_reference = _inter_lane_reference[:,topk_idx,:,:]#layer, topk, 8, 3
                _all_reference = torch.cat((_init_reference.unsqueeze(0), _inter_reference), dim=0)
                query_list.append(_queries)

                ref_pts_list.append(_all_reference)
                lane_ref_pts_list.append(_inter_lane_reference)

            self.query_memory.update(query_list, img_metas)
            self._topk_indices = topk_idx_list
            self._topk_valid = topk_valid_list
            self.reference_points_memory.update(ref_pts_list, img_metas)
            self.lane_ref_points_memory.update(lane_ref_pts_list, img_metas)

        return outs

    def _get_target_single(self,
                           cls_score,
                           lanes_pred,
                           masks_pred,
                           lanes_left_type_preds,
                           lanes_right_type_preds,
                           gt_labels,
                           gt_lanes,
                           gt_instance_masks,
                           gt_lanes_left_type,
                           gt_lanes_right_type,
                           gt_bboxes_ignore=None):

        num_bboxes = lanes_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(lanes_pred, masks_pred, cls_score, gt_lanes,
                                             gt_instance_masks, gt_labels)

        sampling_result = self.sampler.sample(assign_result, lanes_pred, gt_lanes)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds

        labels = gt_lanes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
        label_weights = gt_lanes.new_ones(num_bboxes)

        labels_left_type = gt_lanes_left_type.new_full((num_bboxes,), self.num_lane_type_classes, dtype=torch.long)
        labels_left_type[pos_inds] = gt_lanes_left_type[sampling_result.pos_assigned_gt_inds].long()

        labels_right_type = gt_lanes_right_type.new_full((num_bboxes,), self.num_lane_type_classes, dtype=torch.long)
        labels_right_type[pos_inds] = gt_lanes_right_type[sampling_result.pos_assigned_gt_inds].long()

        # bbox targets
        gt_c = gt_lanes.shape[-1]
        if gt_c == 0:
            gt_c = self.gt_c_save
            sampling_result.pos_gt_bboxes = torch.zeros((0, gt_c)).to(sampling_result.pos_gt_bboxes.device)
        else:
            self.gt_c_save = gt_c

        bbox_targets = torch.zeros_like(lanes_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(lanes_pred)
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        bbox_weights[pos_inds] = 1.0

        # mask targets
        mask_targets = gt_instance_masks[pos_assigned_gt_inds]
        mask_weights = masks_pred.new_zeros((self.num_query, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights, mask_targets, mask_weights, labels_left_type, labels_right_type,
                pos_inds, neg_inds, pos_assigned_gt_inds)

    def get_targets(self,
                    cls_scores_list,
                    lanes_preds_list,
                    masks_preds_list,
                    lanes_left_type_preds_list,
                    lanes_right_type_preds_list,
                    gt_lanes_list,
                    gt_labels_list,
                    gt_masks_list,
                    gt_lanes_left_type_list,
                    gt_lanes_right_type_list,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, lanes_targets_list,
         lanes_weights_list, masks_targets_list, masks_weights_list, labels_left_type_list, labels_right_type_list,
         pos_inds_list, neg_inds_list, pos_assigned_gt_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, lanes_preds_list, masks_preds_list, lanes_left_type_preds_list, lanes_right_type_preds_list,
            gt_labels_list, gt_lanes_list, gt_masks_list, gt_lanes_left_type_list, gt_lanes_right_type_list, gt_bboxes_ignore_list)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        assign_result = dict(
            pos_inds=pos_inds_list, neg_inds=neg_inds_list, pos_assigned_gt_inds=pos_assigned_gt_inds_list
        )

        return (labels_list, label_weights_list, lanes_targets_list,
                lanes_weights_list, masks_targets_list, masks_weights_list, labels_left_type_list, labels_right_type_list,
                num_total_pos, num_total_neg, assign_result)

    def loss_single(self,
                    cls_scores,
                    lanes_preds,
                    masks_preds,
                    lanes_left_type,
                    lanes_right_type,
                    gt_lanes_list,
                    gt_labels_list,
                    gt_masks_list,
                    gt_lanes_left_type,
                    gt_lanes_right_type,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        lanes_preds_list = [lanes_preds[i] for i in range(num_imgs)]
        mask_preds_list = [masks_preds[i] for i in range(num_imgs)]
        lanes_left_type_list = [lanes_left_type[i] for i in range(num_imgs)]
        lanes_right_type_list = [lanes_right_type[i] for i in range(num_imgs)]
        cls_reg_seg_targets = self.get_targets(cls_scores_list, lanes_preds_list, mask_preds_list, lanes_left_type_list, lanes_right_type_list,
                                           gt_lanes_list, gt_labels_list, gt_masks_list, gt_lanes_left_type, gt_lanes_right_type,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, mask_targets_list, mask_weights_list, labels_left_type_list, labels_right_type_list,
         num_total_pos, num_total_neg, assign_result) = cls_reg_seg_targets

        labels = torch.cat(labels_list, 0) #torch.Size([400])
        labels_left_type = torch.cat(labels_left_type_list, 0) #torch.Size([400])
        labels_right_type = torch.cat(labels_right_type_list, 0) #torch.Size([400])
        label_weights = torch.cat(label_weights_list, 0) #torch.Size([400])
        bbox_targets = torch.cat(bbox_targets_list, 0) #torch.Size([400, 90])
        bbox_weights = torch.cat(bbox_weights_list, 0) #torch.Size([400, 90])
        mask_targets = torch.cat(mask_targets_list, 0) #torch.Size([66, 100, 200])
        mask_weights = torch.stack(mask_weights_list, 0) #torch.Size([2, 200])

        mask_targets_out = mask_targets
        # classification loss

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        lanes_left_type = lanes_left_type.reshape(-1, self.cls_lane_type_out_channels) #torch.Size([2, 200, 3]) -> torch.Size([2, 200, 3])
        lanes_right_type = lanes_right_type.reshape(-1, self.cls_lane_type_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        ###cls_scores torch.Size([400, 2]) torch.float32
        ###labels torch.Size([400]) torch.int64
        ##label_weights torch.Size([400]) torch.float32

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        ###lanes_left_type torch.Size([400, 3]) torch.float32
        ###lanes_left_type torch.Size([400]) torch.int64
        # cls_avg_factor 53.0 float
        loss_left_type = self.loss_lane_type(
            lanes_left_type, labels_left_type, label_weights, avg_factor=cls_avg_factor)

        loss_right_type = self.loss_lane_type(
            lanes_right_type, labels_right_type, label_weights, avg_factor=cls_avg_factor)

        loss_lane_type = loss_left_type + loss_right_type
        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        lanes_preds = lanes_preds.reshape(-1, lanes_preds.size(-1))

        isnotnan = torch.isfinite(bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            lanes_preds[isnotnan, :self.code_size],
            bbox_targets[isnotnan, :self.code_size],
            bbox_weights[isnotnan, :self.code_size],
            avg_factor=num_total_pos)
        #

        # segmentation part
        cls_scores = cls_scores.flatten(0,1)
        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones, shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        ## masks_preds torch.Size([2, 200, 100, 200])
        mask_preds = masks_preds[mask_weights > 0] #torch.Size([66, 100, 200])

        if mask_targets.shape[0] == 0:
            loss_dice = masks_preds.sum()
            loss_mask = masks_preds.sum()
            return loss_cls, loss_bbox, loss_dice, loss_mask, assign_result

        ### mask_preds torch.Size([66, 100, 200])
        ## mask_targets torch.Size([66, 100, 200])
        ## num_total_masks [66.]
        # dice loss mask_preds
        loss_dice = self.loss_dice(
            mask_preds, mask_targets, avg_factor=num_total_masks
        )

        # mask loss (point based, deprecated)
        # FocalLoss support input of shape (n, num_class)
        h, w = mask_preds.shape[-2:]
        # shape (num_total_gts, h, w) -> (num_total_gts * h * w, 1)
        mask_preds = mask_preds.reshape(-1, 1)
        # shape (num_total_gts, h, w) -> (num_total_gts * h * w)
        mask_targets = mask_targets.reshape(-1)
        if self.loss_mask_type == 'FocalLoss':
            mask_targets = (1 - mask_targets).long()
        if self.loss_mask_type == 'CrossEntropyLoss':
            mask_targets = mask_targets.reshape(mask_preds.shape).bool()

        ##mask_preds torch.Size([1320000, 1]) 66x100x200
        ##mask_targets torch.Size([1320000, 1]) 66x100x200
        loss_mask = self.loss_mask(
            mask_preds, mask_targets, avg_factor=num_total_masks * h * w
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_dice = torch.nan_to_num(loss_dice)
            loss_mask = torch.nan_to_num(loss_mask)

        return loss_cls, loss_bbox, loss_dice, loss_mask, loss_lane_type, assign_result, bbox_targets, labels, labels_left_type, labels_right_type, mask_targets_out, mask_weights

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             preds_dicts,
             gt_lanes_3d,
             gt_labels_list,
             gt_instance_masks,
             gt_lane_left_type,
             gt_lane_right_type,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_instance_masks (list[Tensor]): Ground truth instance masks for each lane segment
                of map size with shape (num_gts, 100, 50)
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_lanes_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                all_masks_preds (Tensor): Bitwise instance segmentation outputs of
                    all decoder layers. Each is a bitwise segmentation map with shape (100,50).
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_lanes_preds = preds_dicts['all_lanes_preds']
        all_mask_preds = preds_dicts['all_mask_preds']
        all_lanes_left_type = preds_dicts['all_lanes_left_type']
        all_lanes_right_type = preds_dicts['all_lanes_right_type']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_lanes_list = [lane for lane in gt_lanes_3d]

        all_gt_lanes_list = [gt_lanes_list for _ in range(num_dec_layers)]

        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        gt_masks_list = [mask for mask in gt_instance_masks]

        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]

        all_gt_lanes_left_type = [gt_lane_left_type for _ in range(num_dec_layers)]

        all_gt_lanes_right_type = [gt_lane_right_type for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_dice, losses_mask, losses_lane_type, assign_result, bbox_targets, labels_targets, labels_left_type_targets, labels_right_type_targets, mask_targets, mask_targets_weights = multi_apply(
            self.loss_single, all_cls_scores, all_lanes_preds, all_mask_preds, all_lanes_left_type, all_lanes_right_type,
            all_gt_lanes_list, all_gt_labels_list, all_gt_masks_list, all_gt_lanes_left_type, all_gt_lanes_right_type)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_lane_cls'] = losses_cls[-1]
        loss_dict['loss_lane_reg'] = losses_bbox[-1]
        loss_dict['loss_seg_dice'] = losses_dice[-1]
        loss_dict['loss_seg_mask'] = losses_mask[-1]
        loss_dict['loss_lane_type'] = losses_lane_type[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_dice_i, loss_mask_i, loss_lane_type_i in zip(losses_cls[:-1],
                                                                                       losses_bbox[:-1],
                                                                                       losses_dice[:-1],
                                                                                       losses_mask[:-1],
                                                                                       losses_lane_type[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_lane_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_lane_reg'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_seg_dice'] = loss_dice_i
            loss_dict[f'd{num_dec_layer}.loss_seg_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_lane_type'] = loss_lane_type_i
            num_dec_layer += 1

        return loss_dict, assign_result, bbox_targets, labels_targets, labels_left_type_targets, labels_right_type_targets, mask_targets, mask_targets_weights

    def loss_dn_single(self,
                    cls_scores,
                    lanes_preds,
                    masks_preds,
                    lanes_left_type,
                    lanes_right_type,
                    gt_lanes_list,
                    gt_labels_list,
                    gt_masks_list,
                    gt_lanes_left_type,
                    gt_lanes_right_type,
                    dn_meta,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)

        group = dn_meta['num_dn_group']
        labels = torch.cat(gt_labels_list, 0).repeat(group)
        labels_left_type = torch.cat(gt_lanes_left_type, 0).repeat(group)
        labels_right_type = torch.cat(gt_lanes_right_type, 0).repeat(group)

        num_first = torch.sum(torch.eq(dn_meta['known_bid'],0))

        label_weights = dn_meta['loss_weight']
        bbox_targets = torch.cat(gt_lanes_list, dim=0).repeat(group,1)
        bbox_weights = dn_meta['loss_weight'][:, None].repeat(1, lanes_preds.size(2))
        mask_targets = torch.cat(gt_masks_list, dim=0).repeat(group,1,1)

        num_total_pos = dn_meta['loss_weight'].sum().item()
        cls_avg_factor = num_total_pos * 1.0
        # classification loss

        # construct weighted avg_factor to match with the official DETR repo
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)

        cls_scores = cls_scores[(dn_meta['known_bid'].long(), dn_meta['map_known_indice'].long())]
        lanes_left_type = lanes_left_type[(dn_meta['known_bid'].long(), dn_meta['map_known_indice'].long())]
        lanes_right_type = lanes_right_type[(dn_meta['known_bid'].long(), dn_meta['map_known_indice'].long())]

        loss_cls = self.loss_dn_cls(
            cls_scores, labels, label_weights.float(), avg_factor=cls_avg_factor)

        loss_left_type = self.loss_dn_lane_type(
            lanes_left_type, labels_left_type, label_weights.float(), avg_factor=cls_avg_factor)

        loss_right_type = self.loss_dn_lane_type(
            lanes_right_type, labels_right_type, label_weights.float(), avg_factor=cls_avg_factor)

        loss_lane_type = loss_left_type + loss_right_type
        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss

        lanes_preds = lanes_preds[(dn_meta['known_bid'].long(), dn_meta['map_known_indice'].long())]
        isnotnan = torch.isfinite(bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_dn_bbox(
            lanes_preds[isnotnan, :self.code_size],
            bbox_targets[isnotnan, :self.code_size],
            bbox_weights[isnotnan, :self.code_size],
            avg_factor=num_total_pos)

        # mmcv.imwrite(conn_img, output_path)

        # segmentation part
        cls_scores = cls_scores.flatten(0,1)
        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones, shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)

        mask_preds = masks_preds[(dn_meta['known_bid'].long(), dn_meta['map_known_indice'].long())]

        #     return loss_cls, loss_bbox, loss_dice, loss_mask, assign_result

        # dice loss
        loss_dice = self.loss_dn_dice(
            mask_preds, mask_targets, avg_factor=num_total_masks
        )

        # mask loss (point based, deprecated)
        # FocalLoss support input of shape (n, num_class)
        h, w = mask_preds.shape[-2:]
        # shape (num_total_gts, h, w) -> (num_total_gts * h * w, 1)
        mask_preds = mask_preds.reshape(-1, 1)
        # shape (num_total_gts, h, w) -> (num_total_gts * h * w)
        mask_targets = mask_targets.reshape(-1)
        if self.loss_mask_type == 'FocalLoss':
            mask_targets = (1 - mask_targets).long()
        if self.loss_mask_type == 'CrossEntropyLoss':
            mask_targets = mask_targets.reshape(mask_preds.shape).bool()

        loss_mask = self.loss_dn_mask(
            mask_preds, mask_targets, avg_factor=num_total_masks * h * w
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_dice = torch.nan_to_num(loss_dice)
            loss_mask = torch.nan_to_num(loss_mask)

        loss_dict = dict(
            dn_cls=loss_cls,
            dn_bbox=loss_bbox,
            dn_dice=loss_dice,
            dn_mask=loss_mask,
            dn_lane_type=loss_lane_type,
        )

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def loss_dn(self,
             preds_dicts,
             gt_lanes_3d,
             gt_labels_list,
             gt_instance_masks,
             gt_lane_left_type,
             gt_lane_right_type,
             dn_meta,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_instance_masks (list[Tensor]): Ground truth instance masks for each lane segment
                of map size with shape (num_gts, 100, 50)
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_lanes_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                all_masks_preds (Tensor): Bitwise instance segmentation outputs of
                    all decoder layers. Each is a bitwise segmentation map with shape (100,50).
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_lanes_preds = preds_dicts['all_lanes_preds']
        all_mask_preds = preds_dicts['all_mask_preds']
        all_lanes_left_type = preds_dicts['all_lanes_left_type']
        all_lanes_right_type = preds_dicts['all_lanes_right_type']
        losses = []
        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_lanes_list = [lane for lane in gt_lanes_3d]

        all_gt_lanes_list = [gt_lanes_list for _ in range(num_dec_layers)]

        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        gt_masks_list = [mask for mask in gt_instance_masks]

        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]

        all_gt_lanes_left_type = [gt_lane_left_type for _ in range(num_dec_layers)]

        all_gt_lanes_right_type = [gt_lane_right_type for _ in range(num_dec_layers)]

        dn_meta['loss_weight'] = torch.cat(dn_meta['loss_weight']).repeat(dn_meta['num_dn_group'])

        for i in range(num_dec_layers):
            losses.append(self.loss_dn_single(all_cls_scores[i], all_lanes_preds[i], all_mask_preds[i], all_lanes_left_type[i], all_lanes_right_type[i],
            all_gt_lanes_list[i], all_gt_labels_list[i], all_gt_masks_list[i], all_gt_lanes_left_type[i], all_gt_lanes_right_type[i], dn_meta))
        # losses_cls, losses_bbox, losses_dice, losses_mask, losses_lane_type, assign_result, bbox_targets = multi_apply(

        #     all_gt_lanes_list, all_gt_labels_list, all_gt_masks_list, all_gt_lanes_left_type, all_gt_lanes_right_type)

        loss_dict = dict()
        # loss from the last decoder layer

        loss_dict['loss_dn_lane_cls'] = losses[-1]['dn_cls']
        loss_dict['loss_dn_lane_reg'] = losses[-1]['dn_bbox']
        # loss_dict['loss_dn_seg_dice'] = losses[-1]['dn_dice']
        # loss_dict['loss_dn_seg_mask'] = losses[-1]['dn_mask']
        loss_dict['loss_dn_lane_type'] = losses[-1]['dn_lane_type']

        losses_cls = [loss['dn_cls'] for loss in losses]
        losses_bbox = [loss['dn_bbox'] for loss in losses]
        losses_dice = [loss['dn_dice'] for loss in losses]
        losses_mask = [loss['dn_mask'] for loss in losses]
        losses_lane_type = [loss['dn_lane_type'] for loss in losses]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_dice_i, loss_mask_i, loss_lane_type_i in zip(losses_cls[:-1],
                                                                                       losses_bbox[:-1],
                                                                                       losses_dice[:-1],
                                                                                       losses_mask[:-1],
                                                                                       losses_lane_type[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_dn_lane_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_dn_lane_reg'] = loss_bbox_i
            # loss_dict[f'd{num_dec_layer}.loss_dn_seg_dice'] = loss_dice_i
            # loss_dict[f'd{num_dec_layer}.loss_dn_seg_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dn_lane_type'] = loss_lane_type_i
            num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_lanes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            lanes = preds['lane3d']
            scores = preds['scores']
            labels = preds['labels']
            result = [lanes, scores, labels]
            if 'left_type_scores' in preds:
                left_type_scores = preds['left_type_scores']
                left_type_labels = preds['left_type_labels']
                right_type_scores = preds['right_type_scores']
                right_type_labels = preds['right_type_labels']
                result.extend([left_type_scores, left_type_labels, right_type_scores, right_type_labels])
            ret_list.append(result)

        return ret_list

    def store_topology_memory(self, relationship_probability, img_metas):
        """Store the same ordered Top-K subgraph as the node memories."""
        if not self.topology_memory_enabled:
            return
        if relationship_probability.ndim != 3 or len(self._topk_indices) != relationship_probability.shape[0]:
            raise ValueError('Expected one directed adjacency matrix per sample')
        selected = []
        for batch_idx, indices in enumerate(self._topk_indices):
            adjacency = relationship_probability[batch_idx]
            if adjacency.shape[0] != adjacency.shape[1] or indices.numel() != self.topk_query:
                raise ValueError('Invalid relationship or Top-K dimensions')
            selected.append(adjacency.index_select(0, indices).index_select(1, indices))
        self.topology_memory.update(selected, img_metas)
        self.topology_valid_memory.update(self._topk_valid, img_metas)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        for k, v in self.__dict__.items():
            if isinstance(v, StreamTensorMemory):
                v.train(*args, **kwargs)

    def eval(self):
        super().eval()
        for k, v in self.__dict__.items():
            if isinstance(v, StreamTensorMemory):
                v.eval()

    def forward(self, *args, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def prepare_temporal_propagation(self, preds_dict, scene_name, local_idx, memory_bank=None,
                        thr_track=0.1, thr_det=0.5):
        lines = preds_dict['all_lanes_preds'][0] # List[Tensor(num_queries, 2*num_points)]
        queries = preds_dict['history_states'][0]

        left_types = preds_dict['all_lanes_left_type'][0]
        right_types = preds_dict['all_lanes_right_type'][0]
        bs = len(lines)
        assert bs == 1, 'now only support bs=1 for temporal-evolving inference'
        scores = preds_dict['all_cls_scores'][0] # (bs, num_queries, 3)

        first_frame = local_idx == 0

        tmp_vectors = lines[0]
        tmp_queries = queries[0]

        tmp_left_types = left_types[0]
        tmp_right_types = right_types[0]

        # focal loss
        if self.loss_cls.use_sigmoid:
            tmp_scores, tmp_labels = scores[0].max(-1)
            tmp_scores = tmp_scores.sigmoid()

            if not first_frame:

                last_track_num = self.prop_info['prop_frame_num_instance'] if self.prop_info['prop_frame_num_instance']<= self.topk_query else self.topk_query
                pos_track = tmp_scores[:last_track_num] > thr_track
                pos_det = tmp_scores[last_track_num:] > thr_det
                pos = torch.cat([pos_track, pos_det], dim=0)
                pass  # Keep inference silent.
            else:
                pos_det = tmp_scores > thr_det
                pos = pos_det
                pass  # Keep inference silent.

        else:
            raise RuntimeError('The experiment uses sigmoid for cls outputs')

        if torch.sum(pos)> self.topk_query and not first_frame:
            _, valid_idx = torch.topk(tmp_scores[last_track_num:], k=self.topk_query - torch.sum(pos_track), dim=-1)
            pad_pos_det = torch.zeros_like(pos_det, dtype=torch.bool)
            pad_pos_det[valid_idx] = True
            pos_det = pad_pos_det
            pos = torch.cat([pos_track, pos_det], dim=0)
        elif torch.sum(pos)> self.topk_query and first_frame:
            _, valid_idx = torch.topk(tmp_scores, k=self.topk_query , dim=-1)
            pad_pos_det = torch.zeros_like(pos_det, dtype=torch.bool)
            pad_pos_det[valid_idx] = True
            pos_det = pad_pos_det
            pos = pos_det

        pos_vectors = tmp_vectors[pos]
        pos_labels = tmp_labels[pos]
        pos_queries = tmp_queries[pos]
        pos_scores = tmp_scores[pos]

        _, pos_left_types = tmp_left_types[pos].sigmoid().max(-1)
        _, pos_right_types = tmp_right_types[pos].sigmoid().max(-1)

        if first_frame:
            global_ids = torch.arange(len(pos_vectors))
            num_instance = len(pos_vectors)

        else:

            prop_ids = self.prop_info['global_ids']
            prop_num_instance = self.prop_info['num_instance']
            prop_frame_num_instance = self.prop_info['prop_frame_num_instance']

            try:
                global_ids_track = prop_ids[pos_track[:len(prop_ids)]]
            except:
                raise  # preserve original exception
            num_newborn = int(pos_det.sum())
            global_ids_newborn = torch.arange(num_newborn) + prop_num_instance
            pass  # Keep inference silent.
            global_ids = torch.cat([global_ids_track, global_ids_newborn])
            pass  # Keep inference silent.
            num_instance = prop_num_instance + num_newborn

        if first_frame:
            self.prop_info = {
                'vectors': pos_vectors,
                'queries': pos_queries,
                'scores': pos_scores,
                'labels': pos_labels,
                # 'masks': pos_masks,
                'left_types': pos_left_types,
                'right_types': pos_right_types,
                'scene_name': scene_name,
                'local_idx': local_idx,
                'global_ids': global_ids,
                'num_instance': num_instance,
                'prop_frame_num_instance':num_instance,
            }
        else:
            self.prop_info = {
                'vectors': pos_vectors,
                'queries': pos_queries,
                'scores': pos_scores,
                'labels': pos_labels,
                # 'masks': pos_masks,
                'left_types': pos_left_types,
                'right_types': pos_right_types,
                'scene_name': scene_name,
                'local_idx': local_idx,
                'global_ids': global_ids,
                'num_instance': num_instance,
                'prop_frame_num_instance':len(pos_vectors),
            }

        if memory_bank is not None:
            if first_frame:
                num_tracks = 0
            else:
                num_tracks = self.prop_active_tracks
            pos_out_inds = torch.where(pos)[0]
            prev_out = {
                'history_states': queries,
                'all_cls_scores': scores,
            }

            memory_bank.update_memory(0, first_frame, pos_out_inds, prev_out, num_tracks, local_idx, memory_bank.curr_t)
            self.prop_active_tracks = len(pos_out_inds)

        save_pos_results = {
            'vectors': pos_vectors.cpu().numpy(),
            'scores': pos_scores.cpu().numpy(),
            'labels': pos_labels.cpu().numpy(),
            # 'masks': pos_masks.cpu().numpy(),
            'left_types': pos_left_types.cpu().numpy(),
            'right_types': pos_right_types.cpu().numpy(),
            'global_ids': global_ids.cpu().numpy(),
            'scene_name': scene_name,
            'local_idx': local_idx,
            'num_instance': num_instance,
        }

        return [save_pos_results]
