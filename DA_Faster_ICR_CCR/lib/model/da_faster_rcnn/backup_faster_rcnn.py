import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.da_faster_rcnn.DA import _ImageDA, _InstanceDA
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.rpn import _RPN
from model.utils.config import cfg
from model.utils.net_utils import (
    _affine_grid_gen,
    _affine_theta,
    _crop_pool_layer,
    _smooth_l1_loss,
)
from torch.autograd import Variable


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic, in_channel=4096):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign(
            (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0
        )

        self.grid_size = (
            cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        )
        # self.RCNN_roi_crop = _RoICrop()

        self.RCNN_imageDA = _ImageDA(self.dout_base_model)
        self.RCNN_instanceDA = _InstanceDA(in_channel)
        self.consistency_loss = torch.nn.MSELoss(size_average=False)
        self.conv_lst = nn.Conv2d(self.dout_base_model, self.n_classes - 1, 1, 1, 0)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(
        self,
        im_data,
        im_info,
        im_cls_lb,
        gt_boxes,
        num_boxes,
        need_backprop,
        tgt_im_data,
        tgt_im_info,
        tgt_gt_boxes,
        tgt_num_boxes,
        tgt_need_backprop,
    ):

        if not (need_backprop.detach() == 1 and tgt_need_backprop.detach() == 0):
            need_backprop = torch.Tensor([1]).cuda()
            tgt_need_backprop = torch.Tensor([0]).cuda()

        assert need_backprop.detach() == 1 and tgt_need_backprop.detach() == 0

        batch_size = im_data.size(0)
        im_info = im_info.data  # (size1,size2, image ratio(new image / source image) )
        im_cls_lb = im_cls_lb.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        need_backprop = need_backprop.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        cls_feat = self.conv_lst(self.avg_pool(base_feat)).squeeze(-1).squeeze(-1)
        img_cls_loss = nn.BCEWithLogitsLoss()(cls_feat, im_cls_lb)

        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.train()
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(
            base_feat, im_info, gt_boxes, num_boxes
        )

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(
                rois_outside_ws.view(-1, rois_outside_ws.size(2))
            )

        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == "align":
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == "pool":
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(
                bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4
            )
            bbox_pred_select = torch.gather(
                bbox_pred_view,
                1,
                rois_label.view(rois_label.size(0), 1, 1).expand(
                    rois_label.size(0), 1, 4
                ),
            )
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(
                bbox_pred, rois_target, rois_inside_ws, rois_outside_ws
            )

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        """ =================== for target =========================="""

        tgt_batch_size = tgt_im_data.size(0)
        tgt_im_info = (
            tgt_im_info.data
        )  # (size1,size2, image ratio(new image / source image) )
        tgt_gt_boxes = tgt_gt_boxes.data
        tgt_num_boxes = tgt_num_boxes.data
        tgt_need_backprop = tgt_need_backprop.data

        # feed image data to base model to obtain base feature map
        tgt_base_feat = self.RCNN_base(tgt_im_data)

        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.eval()
        tgt_rois, tgt_rpn_loss_cls, tgt_rpn_loss_bbox = self.RCNN_rpn(
            tgt_base_feat, tgt_im_info, tgt_gt_boxes, tgt_num_boxes
        )

        # if it is training phrase, then use ground trubut bboxes for refining

        tgt_rois_label = None
        tgt_rois_target = None
        tgt_rois_inside_ws = None
        tgt_rois_outside_ws = None
        tgt_rpn_loss_cls = 0
        tgt_rpn_loss_bbox = 0

        tgt_rois = Variable(tgt_rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == "crop":
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            tgt_grid_xy = _affine_grid_gen(
                tgt_rois.view(-1, 5), tgt_base_feat.size()[2:], self.grid_size
            )
            tgt_grid_yx = torch.stack(
                [tgt_grid_xy.data[:, :, :, 1], tgt_grid_xy.data[:, :, :, 0]], 3
            ).contiguous()
            tgt_pooled_feat = self.RCNN_roi_crop(
                tgt_base_feat, Variable(tgt_grid_yx).detach()
            )
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                tgt_pooled_feat = F.max_pool2d(tgt_pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == "align":
            tgt_pooled_feat = self.RCNN_roi_align(tgt_base_feat, tgt_rois.view(-1, 5))
        elif cfg.POOLING_MODE == "pool":
            tgt_pooled_feat = self.RCNN_roi_pool(tgt_base_feat, tgt_rois.view(-1, 5))

        # feed pooled features to top model
        tgt_pooled_feat = self._head_to_tail(tgt_pooled_feat)

        if tgt_pooled_feat.shape[0] > pooled_feat.shape[0]:
            tgt_pooled_feat = tgt_pooled_feat[: pooled_feat.shape[0]]
        """  DA loss   """

        # DA LOSS
        DA_img_loss_cls = 0
        DA_ins_loss_cls = 0

        tgt_DA_img_loss_cls = 0
        tgt_DA_ins_loss_cls = 0

        base_score, base_label = self.RCNN_imageDA(base_feat, need_backprop)

        # Image DA
        base_prob = F.log_softmax(base_score, dim=1)
        DA_img_loss_cls = F.nll_loss(base_prob, base_label)

        instance_sigmoid, same_size_label = self.RCNN_instanceDA(
            pooled_feat, need_backprop
        )
        instance_loss = nn.BCELoss()
        DA_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)

        # consistency_prob = torch.max(F.softmax(base_score, dim=1),dim=1)[0]
        consistency_prob = F.softmax(base_score, dim=1)[:, 1, :, :]
        consistency_prob = torch.mean(consistency_prob)
        consistency_prob = consistency_prob.repeat(instance_sigmoid.size())

        DA_cst_loss = self.consistency_loss(instance_sigmoid, consistency_prob.detach())

        """  ************** taget loss ****************  """

        tgt_base_score, tgt_base_label = self.RCNN_imageDA(
            tgt_base_feat, tgt_need_backprop
        )

        # Image DA
        tgt_base_prob = F.log_softmax(tgt_base_score, dim=1)
        tgt_DA_img_loss_cls = F.nll_loss(tgt_base_prob, tgt_base_label)

        tgt_instance_sigmoid, tgt_same_size_label = self.RCNN_instanceDA(
            tgt_pooled_feat, tgt_need_backprop
        )
        tgt_instance_loss = nn.BCELoss()

        tgt_DA_ins_loss_cls = tgt_instance_loss(
            tgt_instance_sigmoid, tgt_same_size_label
        )

        tgt_consistency_prob = F.softmax(tgt_base_score, dim=1)[:, 0, :, :]
        tgt_consistency_prob = torch.mean(tgt_consistency_prob)
        tgt_consistency_prob = tgt_consistency_prob.repeat(tgt_instance_sigmoid.size())

        tgt_DA_cst_loss = self.consistency_loss(
            tgt_instance_sigmoid, tgt_consistency_prob.detach()
        )

        return (
            rois,
            cls_prob,
            bbox_pred,
            img_cls_loss,
            rpn_loss_cls,
            rpn_loss_bbox,
            RCNN_loss_cls,
            RCNN_loss_bbox,
            rois_label,
            DA_img_loss_cls,
            DA_ins_loss_cls,
            tgt_DA_img_loss_cls,
            tgt_DA_ins_loss_cls,
            DA_cst_loss,
            tgt_DA_cst_loss,
        )

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean
                )  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.conv_lst, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_imageDA.Conv1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_imageDA.Conv2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_instanceDA.dc_ip1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_instanceDA.dc_ip2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_instanceDA.clssifer, 0, 0.05, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
