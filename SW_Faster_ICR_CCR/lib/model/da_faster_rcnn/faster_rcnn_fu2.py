import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.rpn import _RPN
from model.utils.config import cfg
from model.utils.net_utils import (
    _affine_grid_gen,
    _affine_theta,
    _crop_pool_layer,
    _smooth_l1_loss,
    grad_reverse,
)
from torch.autograd import Variable


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic, lc, gc, ce, grl):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.lc = lc
        self.gc = gc
        self.ce = ce
        self.grl = grl
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign(
            (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0
        )

        self.grid_size = (
            cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        )

        self.conv_lst = nn.Conv2d(self.dout_base_model, self.n_classes - 1, 1, 1, 0)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.bn1 = nn.BatchNorm2d(self.dout_base_model, momentum=0.01)
        # self.bn2 = nn.BatchNorm2d(self.n_classes-1, momentum=0.01)

    # Megvii
    # def forward(
    #     self, im_data, im_info, im_cls_lb, gt_boxes, num_boxes, target=False, eta=1.0
    # ):
    def forward(
            self, im_data, im_info, gt_boxes, num_boxes, target=False, eta=1.0
    ):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)

        if self.lc:
            d_pixel, _ = self.netD_pixel(grad_reverse(base_feat1, lambd=eta))
            # print(d_pixel.shape)  # torch.Size([45000, 1])
            if not target:
                _, feat_pixel = self.netD_pixel(base_feat1.detach())
        else:  # TODO: CHECK lc를 안쓰기 위해 막아놓음
            d_pixel = self.netD_pixel(grad_reverse(base_feat1, lambd=eta))
            # print(d_pixel.shape) #torch.Size([45000, 1])
        base_feat = self.RCNN_base2(base_feat1)
        # print(base_feat.shape)# torch.Size([1, 512, 37, 75])

        # print(d_pixel.shape)#torch.Size([45000, 1])
        if self.ce:
            d_pixel, _ = self.netD_pixel_CE(grad_reverse(base_feat1, lambd=eta))
            # d_pixel = self.netD_pixel_CE(grad_reverse(base_feat1, lambd=eta))
            # print(d_pixel.shape) #torch.Size([1, 2])

            if not target:
                _, feat_pixel = self.netD_pixel_CE(base_feat1.detach())
                # feat_pixel = self.netD_pixel_CE(base_feat1.detach())

        if self.gc:
            domain_p, _ = self.netD(grad_reverse(base_feat, lambd=eta))
            if target and not self.grl:
                return d_pixel, domain_p  # , diff
            _, feat = self.netD(base_feat.detach())  # torch.Size([1, 128])

        else:
            domain_p = self.netD(grad_reverse(base_feat, lambd=eta))
            if target:
                return d_pixel, domain_p  # ,diff



        if self.grl:
            # projection = torch.sum(base_feat, dim=2).unsqueeze(dim=2)
            # print(projection.shape)
            # print("base_feat",base_feat.shape)
            projection = torch.sum(base_feat, dim=3).unsqueeze(dim=3)
            # print("projection",projection.shape)
            # exit()
            d_pixel_grl, _ = self.netD_pixel(grad_reverse(base_feat1, lambd=eta))
            projection_grl,_ = self.netD_grl(grad_reverse(projection, lambd=eta)) #context: True
            # projection_grl = self.netD_grl(grad_reverse(projection, lambd=eta))  # context: False
            if target and self.gc:
                return d_pixel, domain_p, d_pixel_grl, projection_grl
            _, feat_grl = self.netD_grl(projection.detach())
        # else:
        #     # projection_grl = self.netD(grad_reverse(base_feat, lambd=eta))
        #     # if target:
        #     #     return d_pixel, projection_grl  ##TODO: CHECK d_pixel_grl
        #     domain_p = self.netD(grad_reverse(base_feat, lambd=eta))
        #     if target:
        #         return d_pixel, domain_p  # ,diff

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(
            base_feat, im_info, gt_boxes, num_boxes
        )
        # supervise base feature map with category level label
        # cls_feat = self.avg_pool(base_feat) #Megvii
        # cls_feat = self.conv_lst(cls_feat).squeeze(-1).squeeze(-1) #Megvii
        # cls_feat = self.conv_lst(self.bn1(self.avg_pool(base_feat))).squeeze(-1).squeeze(-1)
        # category_loss_cls = nn.BCEWithLogitsLoss()(cls_feat, im_cls_lb) #Megvii

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
        # feat_pixel = torch.zeros(feat_pixel.size()).cuda()
        if self.lc:
            feat_pixel = feat_pixel.view(1, -1).repeat(pooled_feat.size(0), 1)
            pooled_feat = torch.cat((feat_pixel, pooled_feat), 1)
        if self.gc:
            feat = feat.view(1, -1).repeat(pooled_feat.size(0), 1)
            pooled_feat = torch.cat((feat, pooled_feat), 1)
            # compute bbox offset
        if self.ce:
            feat_pixel = feat_pixel.view(1, -1).repeat(pooled_feat.size(0), 1)
            pooled_feat = torch.cat((feat_pixel, pooled_feat), 1)


        if self.grl:
            feat_grl = feat_grl.view(1, -1).repeat(pooled_feat.size(0), 1)
            pooled_feat = torch.cat((feat_grl, pooled_feat), 1)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
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

        return_list = [rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox,
                       rois_label, d_pixel, domain_p]
        if self.grl:
            return_list.append(d_pixel_grl)
            return_list.append(projection_grl)
        else:
            return_list.append(None)
            return_list.append(None)

        return return_list
        # return (
        #     rois,
        #     cls_prob,
        #     bbox_pred,
        #     # category_loss_cls, #Megvii
        #     rpn_loss_cls,
        #     rpn_loss_bbox,
        #     RCNN_loss_cls,
        #     RCNN_loss_bbox,
        #     rois_label,
        #     d_pixel,
        #     domain_p,
        #     d_pixel_grl,
        #     projection_grl
        # )  # ,diff

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

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
