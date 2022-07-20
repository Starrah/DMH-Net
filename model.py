import math
import types
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drn import drn_d_22, drn_d_38, drn_d_54
from e2plabel.e2plabelconvert import VIEW_NAME
from layers import FusionHoughStage, PerspectiveE2PP2E, HoughNewUpSampler

ENCODER_RESNET = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                  'resnet_official_34']
ENCODER_DENSENET = ['densenet121', 'densenet169', 'densenet161', 'densenet201']
ENCODER_HOUGH = ['unet18', 'vgg16', 'drn38', 'drn22', 'drn54']


def OfficialResnetWrapper(model):
    # 从torchvision 0.10.0源码的resnet.py中复制
    def _forward_impl(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:  # Tuple[torch.Tensor * 5]
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        c1 = self.relu(x)
        x = self.maxpool(c1)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # x = self.avgpool(x) # 不需要分类系列特征，所以不要GAP和全连接
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return c1, c2, c3, c4, c5

    model._forward_impl = types.MethodType(_forward_impl, model)
    return model


class DMHNet(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, cfg, backbone, use_rnn):
        super(DMHNet, self).__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.use_rnn = use_rnn  # 应该是没用到的参数
        self.out_scale = 4
        self.step_cols = 1
        self.hidden_size = 256
        self.fov = 160

        # Encoder
        def makeFeatureExtractor():
            if backbone == "resnet_official_34":
                from torchvision.models.resnet import resnet34
                return OfficialResnetWrapper(resnet34(pretrained=True))
            if backbone == "resnet_official_50":
                from torchvision.models.resnet import resnet50
                return OfficialResnetWrapper(resnet50(pretrained=True))
            if backbone == "resnet_official_18":
                from torchvision.models.resnet import resnet18
                return OfficialResnetWrapper(resnet18(pretrained=True))
            if backbone == "resnet_official_101":
                from torchvision.models.resnet import resnet101
                return OfficialResnetWrapper(resnet101(pretrained=True))
            elif backbone.startswith('drn22'):
                return drn_d_22(pretrained=True, out_middle=True)
            elif backbone.startswith('drn38'):
                return drn_d_38(pretrained=True, out_middle=True)
            elif backbone.startswith('drn54'):
                return drn_d_54(pretrained=True, out_middle=True)
            else:
                raise NotImplementedError()

        self.feature_extractor = [makeFeatureExtractor()]
        self._feature_extractor_ref = [0] * 7  # 第七个表示全景图所使用的feature_extractor
        if self.cfg.MODEL.BACKBONE.PRIVATE_UPDOWN:
            self.feature_extractor.append(makeFeatureExtractor())
            self._feature_extractor_ref[4:6] = [len(self.feature_extractor) - 1] * 2
            if self.cfg.MODEL.BACKBONE.PRIVATE_UP:
                self.feature_extractor.append(makeFeatureExtractor())
                self._feature_extractor_ref[5] = len(self.feature_extractor) - 1
        self.feature_extractor = nn.ModuleList(self.feature_extractor)

        # Input shape
        H, W = 512, 1024
        # Inference channels number from each block of the encoder
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 512)
            if backbone.startswith('drn'):
                net_out = self.feature_extractor[0](dummy)[1]
            else:
                net_out = self.feature_extractor[0](dummy)
            c0, c1, c2, c3, c4 = [b.shape[1] for b in net_out]
            size0, size1, size2, size3, size4 = [b.shape[2] for b in net_out]
            self.c0, self.c1, self.c2, self.c3, self.c4 = c0, c1, c2, c3, c4
            # print("c0, c1, c2, c3, c4", c0, c1, c2, c3, c4)
            c_last = int((c1 * 8 + c2 * 4 + c3 * 4 + c4 * 4) / self.out_scale)

        self.x_mean.requires_grad = False
        self.x_std.requires_grad = False

        def make5HoughModules():
            return nn.ModuleList([
                PerspectiveE2PP2E(self.cfg, size0, size0, size0, self.fov, c0, 1),
                PerspectiveE2PP2E(self.cfg, size1, size1, size1, self.fov, c1, 1),
                # TODO 对降维到hw=64的特征图，角度的霍夫投票个数还能是180吗？
                PerspectiveE2PP2E(self.cfg, size2, size2, size2, self.fov, c2, 1,
                                  hough_angles_num=90),
                PerspectiveE2PP2E(self.cfg, size3, size3, size3, self.fov, c3, 1,
                                  hough_angles_num=90),
                PerspectiveE2PP2E(self.cfg, size4, size4, size4, self.fov, c4, 1,
                                  hough_angles_num=90),
            ])

        self.hough = [make5HoughModules(), make5HoughModules(), make5HoughModules()]
        self._hough_ref = [0, 0, 0, 0, 1, 2]
        self.hough = nn.ModuleList(self.hough)

        def make2FusionModules():
            factor = self.cfg.MODEL.get("CONV1_CHANNEL_FACTOR", 2)
            return nn.ModuleList([
                FusionHoughStage(self.cfg, "xy", 3, c0 // factor, c1 // factor, c2 // factor, c3 // factor,
                                 c4 // factor,
                                 upsample_rate=[512 // size0, 512 // size1, 512 // size2, 512 // size3,
                                                512 // size4, ]),  # xy hough特征的fusion
                FusionHoughStage(self.cfg, "cupdown", 3, c0 // factor, c1 // factor, c2 // factor, c3 // factor,
                                 c4 // factor,
                                 upsample_rate=[512 // size0, 512 // size1, 512 // size2, 512 // size3, 512 // size4, ],
                                 upsampler_class=HoughNewUpSampler),
                # cupdown hough特征的fusion
            ])

        self.fusion_stage = [make2FusionModules(), make2FusionModules(), make2FusionModules()]
        self._fusion_stage_ref = [0, 0, 0, 0, 1, 2]
        self.fusion_stage = nn.ModuleList(self.fusion_stage)

    def _input_image_normalize(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

    def _get_feature_extractor(self, i):
        return self.feature_extractor[self._feature_extractor_ref[i]]

    def _get_hough(self, i):
        return self.hough[self._hough_ref[i]]

    def _get_fusion_stage(self, i):
        return self.fusion_stage[self._fusion_stage_ref[i]]

    def forward(self, input):
        results_dict = {}

        p_xys = []
        p_cuds = []
        for view_idx in range(input["p_imgs"].shape[1]):  # 对所有sample的每个view做循环
            p_img = self._input_image_normalize(input["p_imgs"][:, view_idx])

            p_conv_list = self._get_feature_extractor(view_idx)(p_img)
            if len(p_conv_list) == 2: p_conv_list = p_conv_list[1]

            p_hough_bin_feat = [hough(onefeat) for onefeat, hough in zip(p_conv_list, self._get_hough(view_idx))]

            fusioner = self._get_fusion_stage(view_idx)
            # Decoder for xy peaks
            p_hough_feat_xy = [f[0] for f in p_hough_bin_feat]
            p_xy = fusioner[0](p_hough_feat_xy)
            p_xys.append(p_xy)
            # 中心线的解码器
            p_hough_feat_cud = [f[1] for f in p_hough_bin_feat]
            p_cud = fusioner[1](p_hough_feat_cud)
            p_cuds.append(p_cud)

        results_dict.update({
            "p_preds_xy": torch.cat(p_xys, 1),
            "p_preds_cud": torch.cat(p_cuds, 1),
        })

        if self.cfg.MODEL.get("NO_CLINE_PRED"):
            results_dict["p_preds_cud"] = torch.ones_like(results_dict["p_preds_cud"]) * -math.inf
        if self.cfg.MODEL.get("NO_HLINE_PRED"):
            results_dict["p_preds_xy"][:, :, :, 1] = torch.ones_like(results_dict["p_preds_xy"][:, :, :, 1]) * -math.inf
        if self.cfg.MODEL.get("NO_VLINE_PRED"):
            results_dict["p_preds_xy"][:, :, :, 0] = torch.ones_like(results_dict["p_preds_xy"][:, :, :, 0]) * -math.inf

        losses = self.calculate_loss(input, results_dict)

        return losses, results_dict

    def calculate_loss(self, input, output):
        device = input["e_img"].device
        xLabels = input["xLabels"].to(device)
        yLabels = input["yLabels"].to(device)
        cUpLabels = input["cUpLabels"].to(device)
        cDownLabels = input["cDownLabels"].to(device)

        losses = {
            "total": 0.0
        }

        # 附加loss：仅供debug使用，不会计入总量！
        losses["extra_xLabels"] = 0.0
        losses["extra_yLabels"] = 0.0
        losses["extra_cUpLabels"] = 0.0
        losses["extra_cDownLabels"] = 0.0

        # 六个perspective loss
        for view_idx in range(output["p_preds_xy"].shape[1]):
            one_loss_x = []
            one_loss_y = []
            one_loss_c_up = []
            one_loss_c_down = []
            for img_idx in range(output["p_preds_xy"].shape[0]):
                the_onepred_xy = output["p_preds_xy"][img_idx, view_idx]
                if not self.cfg.MODEL.get("NO_VLINE_PRED"):
                    the_oneloss_x = F.binary_cross_entropy_with_logits(the_onepred_xy[:, 0], xLabels[img_idx, view_idx])
                    one_loss_x.append(the_oneloss_x)
                if not self.cfg.MODEL.get("NO_HLINE_PRED"):
                    the_oneloss_y = F.binary_cross_entropy_with_logits(the_onepred_xy[:, 1], yLabels[img_idx, view_idx])
                    one_loss_y.append(the_oneloss_y)
                if not self.cfg.MODEL.get("NO_CLINE_PRED"):
                    the_onepred_cud = output["p_preds_cud"][img_idx, view_idx]
                    the_oneloss_c_up = F.binary_cross_entropy_with_logits(the_onepred_cud[:, 0],
                                                                          cUpLabels[img_idx, view_idx])
                    one_loss_c_up.append(the_oneloss_c_up)
                    the_oneloss_c_down = F.binary_cross_entropy_with_logits(the_onepred_cud[:, 1],
                                                                            cDownLabels[img_idx, view_idx])
                    one_loss_c_down.append(the_oneloss_c_down)

            one_loss_x = (torch.stack(one_loss_x) if len(one_loss_x) > 0 else output["p_preds_xy"].new_tensor([])) \
                             .sum() / output["p_preds_xy"].shape[0]  # 求和再除以batchsize，而不是求平均，以保证每个图片对loss的贡献相同
            one_loss_y = (torch.stack(one_loss_y) if len(one_loss_y) > 0 else output["p_preds_xy"].new_tensor([])) \
                             .sum() / output["p_preds_xy"].shape[0]
            one_loss_c_up = (torch.stack(one_loss_c_up) if len(one_loss_c_up) > 0 else output["p_preds_cud"].new_tensor(
                [])).sum() / output["p_preds_xy"].shape[0]
            one_loss_c_down = (torch.stack(one_loss_c_down) if len(one_loss_c_down) > 0 else output[
                "p_preds_cud"].new_tensor([])).sum() / output["p_preds_xy"].shape[0]

            with torch.no_grad():
                losses["extra_xLabels"] += one_loss_x
                losses["extra_yLabels"] += one_loss_y
                losses["extra_cUpLabels"] += one_loss_c_up
                losses["extra_cDownLabels"] += one_loss_c_down

            one_loss = one_loss_x + one_loss_y + one_loss_c_up + one_loss_c_down
            losses["p_" + VIEW_NAME[view_idx]] = one_loss
            losses["total"] += self.cfg.MODEL.get("LOSS", {}).get("ALPHA_PERSPECTIVE", 1.0) * one_loss

        return losses
