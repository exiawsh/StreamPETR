# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean, bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh)
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from projects.mmdet3d_plugin.models.utils.misc import draw_heatmap_gaussian, apply_center_offset, apply_ltrb
from mmdet.core import bbox_overlaps
from mmdet3d.models.utils import clip_sigmoid
import random

@HEADS.register_module()
class FocalHead(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 embed_dims=256,
                 stride=16,
                 use_hybrid_tokens=False,
                 train_ratio=1.0,
                 infer_ratio=1.0,
                 sync_cls_avg_factor=False,
                 loss_cls2d=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_centerness=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),
                 loss_centers2d=dict(type='L1Loss', loss_weight=5.0),
                 train_cfg=dict(
                     assigner2d=dict(
                         type='HungarianAssigner2D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                         centers2d_cost=dict(type='BBox3DL1Cost', weight=1.0))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        if train_cfg:
            assert 'assigner2d' in train_cfg, 'assigner2d should be provided '\
                'when train_cfg is set.'
            assigner2d = train_cfg['assigner2d']

            self.assigner2d = build_assigner(assigner2d)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.stride=stride
        self.use_hybrid_tokens=use_hybrid_tokens
        self.train_ratio=train_ratio
        self.infer_ratio=infer_ratio

        super(FocalHead, self).__init__(num_classes, in_channels, init_cfg = init_cfg)



        self.loss_cls2d = build_loss(loss_cls2d)
        self.loss_bbox2d = build_loss(loss_bbox2d)
        self.loss_iou2d = build_loss(loss_iou2d)
        self.loss_centers2d = build_loss(loss_centers2d)
        self.loss_centerness = build_loss(loss_centerness)

        self._init_layers()

    def _init_layers(self):
        self.cls = nn.Conv2d(self.embed_dims, self.num_classes, kernel_size=1)

        self.shared_reg= nn.Sequential(
                                 nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=(3, 3), padding=1),
                                 nn.GroupNorm(32, num_channels=self.embed_dims),
                                 nn.ReLU(),)

        self.shared_cls = nn.Sequential(
                                 nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=(3, 3), padding=1),
                                 nn.GroupNorm(32, num_channels=self.embed_dims),
                                 nn.ReLU(),)

        self.centerness = nn.Conv2d(self.embed_dims, 1, kernel_size=1)
        self.ltrb = nn.Conv2d(self.embed_dims, 4, kernel_size=1)
        self.center2d = nn.Conv2d(self.embed_dims, 2, kernel_size=1)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls.bias, bias_init)
        nn.init.constant_(self.centerness.bias, bias_init)

    def forward(self, location, **data):
        src = data['img_feats']
        bs, n, c, h, w= src.shape
        num_tokens = n*h*w
        
        # focal sampling
        if self.training:
            if self.use_hybrid_tokens:
                sample_ratio = random.uniform(0.2, 1.0)
            else:
                sample_ratio = self.train_ratio 
            num_sample_tokens = int(num_tokens * sample_ratio)
           
        else:
            sample_ratio = self.infer_ratio
            num_sample_tokens = int(num_tokens * sample_ratio)


        x = src.flatten(0, 1)
        cls_feat = self.shared_cls(x)
        cls = self.cls(cls_feat)
        centerness = self.centerness(cls_feat)
        cls_logits = cls.permute(0,2,3,1).reshape(bs*n,-1,self.num_classes)
        centerness = centerness.permute(0,2,3,1).reshape(bs*n,-1,1)
        pred_bboxes = None
        pred_centers2d = None
        
        reg_feat = self.shared_reg(x)
        ltrb = self.ltrb(reg_feat).permute(0,2,3,1).contiguous()
        ltrb = ltrb.sigmoid()
        centers2d_offset = self.center2d(reg_feat).permute(0,2,3,1).contiguous()

        centers2d = apply_center_offset(location, centers2d_offset)
        bboxes = apply_ltrb(location, ltrb)
            
        pred_bboxes = bboxes.view(bs*n,-1,4)
        pred_centers2d = centers2d.view(bs*n,-1,2)

        cls_score = cls_logits.topk(1, dim=2).values[..., 0].view(bs, -1, 1)

        sample_weight = cls_score.detach().sigmoid() * centerness.detach().view(bs,-1,1).sigmoid()

        _, topk_indexes = torch.topk(sample_weight, num_sample_tokens, dim=1)


        outs = {
                'enc_cls_scores': cls_logits,
                'enc_bbox_preds': pred_bboxes,
                'pred_centers2d': pred_centers2d,
                'centerness':centerness,
                'topk_indexes':topk_indexes,
            }

        return outs
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes2d_list,
             gt_labels2d_list,
             centers2d,
             depths,
             preds_dicts,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
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

        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        pred_centers2d = preds_dicts['pred_centers2d']
        centerness = preds_dicts['centerness']

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        all_gt_bboxes2d_list = [bboxes2d for i in gt_bboxes2d_list for bboxes2d in i]
        all_gt_labels2d_list = [labels2d for i in gt_labels2d_list for labels2d in i]
        all_centers2d_list = [center2d for i in centers2d for center2d in i]
        all_depths_list = [depth for i in depths for depth in i]
        enc_loss_cls, enc_losses_bbox, enc_losses_iou, centers2d_losses, centerness_losses = \
            self.loss_single(enc_cls_scores, enc_bbox_preds, pred_centers2d, centerness,
                                all_gt_bboxes2d_list, all_gt_labels2d_list, all_centers2d_list,
                                all_depths_list, img_metas, gt_bboxes_ignore)
        loss_dict['enc_loss_cls'] = enc_loss_cls
        loss_dict['enc_loss_bbox'] = enc_losses_bbox
        loss_dict['enc_loss_iou'] = enc_losses_iou
        loss_dict['centers2d_losses'] = centers2d_losses
        loss_dict['centerness_losses'] = centerness_losses
    
        return loss_dict


    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pred_centers2d,
                    centerness,
                    gt_bboxes_list,
                    gt_labels_list,
                    all_centers2d_list,
                    all_depths_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        centers2d_preds_list = [pred_centers2d[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, centers2d_preds_list,
                                             gt_bboxes_list, gt_labels_list, all_centers2d_list,
                                             all_depths_list, img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, centers2d_targets_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        centers2d_targets = torch.cat(centers2d_targets_list, 0)


        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        # construct factors used for rescale bboxes
        img_h, img_w, _ = img_metas[0]['pad_shape'][0]

        factors = []

        for bbox_pred in  bbox_preds:
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou2d(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        iou_score = bbox_overlaps(bboxes_gt, bboxes, is_aligned=True).reshape(-1)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls2d(
            cls_scores, (labels, iou_score.detach()), label_weights, avg_factor=cls_avg_factor)
        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

         #centerness BCE loss
        img_shape = [img_metas[0]['pad_shape'][0]] * num_imgs
        (heatmaps, ) = multi_apply(self._get_heatmap_single, all_centers2d_list, gt_bboxes_list, img_shape)

        heatmaps = torch.stack(heatmaps, dim=0)
        centerness = clip_sigmoid(centerness)
        loss_centerness = self.loss_centerness(
                centerness,
                heatmaps.view(num_imgs, -1, 1),
                avg_factor=max(num_total_pos, 1))

        # regression L1 loss
        loss_bbox = self.loss_bbox2d(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)



        pred_centers2d = pred_centers2d.view(-1, 2)
        # centers2d L1 loss
        loss_centers2d = self.loss_centers2d(
            pred_centers2d, centers2d_targets, bbox_weights[:, 0:2], avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou, loss_centers2d, loss_centerness

    def _get_heatmap_single(self, obj_centers2d, obj_bboxes, img_shape):
        img_h, img_w, _ = img_shape
        heatmap = torch.zeros(img_h // self.stride, img_w // self.stride, device=obj_centers2d.device)
        if len(obj_centers2d) != 0:
            l = obj_centers2d[..., 0:1] - obj_bboxes[..., 0:1]
            t = obj_centers2d[..., 1:2] - obj_bboxes[..., 1:2]
            r = obj_bboxes[..., 2:3] - obj_centers2d[..., 0:1]
            b = obj_bboxes[..., 3:4] - obj_centers2d[..., 1:2]
            bound = torch.cat([l, t, r, b], dim=-1)
            radius = torch.ceil(torch.min(bound, dim=-1)[0] / 16)
            radius = torch.clamp(radius, 1.0).cpu().numpy().tolist()
            for center, r in zip(obj_centers2d, radius):
                heatmap = draw_heatmap_gaussian(heatmap, center / 16, radius=int(r), k=1)
        return (heatmap, )

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    centers2d_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    all_centers2d_list,
                    all_depths_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        img_meta = {'pad_shape':img_metas[0]['pad_shape'][0]}
        img_meta_list = [img_meta for _ in range(num_imgs)]
        # print(1)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         centers2d_targets_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list, centers2d_preds_list,
            gt_bboxes_list, gt_labels_list, all_centers2d_list,
            all_depths_list, img_meta_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                centers2d_targets_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pred_centers2d,
                           gt_bboxes,
                           gt_labels,
                           centers2d,
                           depths,
                           img_meta,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indexes for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indexes for each image.
                - neg_inds (Tensor): Sampled negative indexes for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner2d.assign(bbox_pred, cls_score, pred_centers2d, gt_bboxes,
                                               gt_labels, centers2d, img_meta, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
        label_weights = gt_bboxes.new_ones(num_bboxes)


        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['pad_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        #centers2d target
        centers2d_targets = bbox_pred.new_full((num_bboxes, 2), 0.0, dtype=torch.float32)
        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert sampling_result.pos_assigned_gt_inds.numel() == 0
            centers2d_labels = torch.empty_like(gt_bboxes).view(-1, 2)
        else:
            centers2d_labels = centers2d[sampling_result.pos_assigned_gt_inds.long(), :]
        centers2d_labels_normalized = centers2d_labels / factor[:, 0:2]
        centers2d_targets[pos_inds] = centers2d_labels_normalized
        return (labels, label_weights, bbox_targets, bbox_weights, centers2d_targets,
                pos_inds, neg_inds)