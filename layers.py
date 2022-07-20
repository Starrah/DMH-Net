import math

import torch
import torch.nn as nn


class PerspectiveE2PP2E(nn.Module):
    def __init__(self, cfg, input_h, input_w, pers_h, fov, input_feat, output_feat, hough_angles_num=180,
                 hoguh_clines_tole=1.0):
        super(PerspectiveE2PP2E, self).__init__()
        self.cfg = cfg
        self.hoguh_clines_tole = hoguh_clines_tole
        self.hough_angles_num = hough_angles_num
        self.input_h = input_h
        self.input_w = input_w
        self.pers_h = pers_h
        self.fov = fov
        self.input_feat = input_feat

        dim = input_feat // self.cfg.MODEL.get("CONV1_CHANNEL_FACTOR", 2)

        # conv1构建
        self.conv1_x = nn.Sequential(nn.Conv2d(input_feat, dim, kernel_size=(1, 1), padding=(0, 0)),
                                     nn.BatchNorm2d(dim),
                                     nn.ReLU()
                                     )
        self.conv1_cup = nn.Sequential(nn.Conv2d(input_feat, dim, kernel_size=(1, 1), padding=(0, 0)),
                                       nn.BatchNorm2d(dim),
                                       nn.ReLU()
                                       )
        self.conv1_cdown = nn.Sequential(nn.Conv2d(input_feat, dim, kernel_size=(1, 1), padding=(0, 0)),
                                         nn.BatchNorm2d(dim),
                                         nn.ReLU()
                                         )
        self.conv1_y = nn.Sequential(nn.Conv2d(input_feat, dim, kernel_size=(1, 1), padding=(0, 0)),
                                     nn.BatchNorm2d(dim),
                                     nn.ReLU()
                                     )

        # conv2构建
        self.conv2_x = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0)),
                                     nn.BatchNorm2d(dim),
                                     nn.ReLU(),
                                     nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0)),
                                     nn.BatchNorm2d(dim),
                                     nn.ReLU()
                                     )
        self.conv2_cup = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0)),
                                       nn.BatchNorm2d(dim),
                                       nn.ReLU(),
                                       nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0)),
                                       nn.BatchNorm2d(dim),
                                       nn.ReLU()
                                       )
        self.conv2_cdown = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0)),
                                         nn.BatchNorm2d(dim),
                                         nn.ReLU(),
                                         nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0)),
                                         nn.BatchNorm2d(dim),
                                         nn.ReLU()
                                         )
        self.conv2_y = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0)),
                                     nn.BatchNorm2d(dim),
                                     nn.ReLU(),
                                     nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0)),
                                     nn.BatchNorm2d(dim),
                                     nn.ReLU()
                                     )

        # 过中心点的线的霍夫投票所用的矩阵
        self.vote_mask_c_up = None
        self.vote_mask_c_down = None

    def makeVoteMask(self, img_size, device):
        vote_mask_c_up, vote_mask_c_down = self.makeVoteMaskStatic(self.cfg.MODEL.HOUGH.CLINE_TYPE, img_size, device,
                                                                   self.hough_angles_num, self.hoguh_clines_tole)
        # 解决显存爆炸：转为(h*w, 180)的矩阵，与特征做矩阵乘法
        self.vote_mask_c_up = vote_mask_c_up.reshape(-1, vote_mask_c_up.shape[-1])
        self.vote_mask_c_down = vote_mask_c_down.reshape(-1, vote_mask_c_down.shape[-1])

    @staticmethod
    def makeVoteMaskStatic(type: str, img_size, device, hough_angles_num=180, hoguh_clines_tole=1.0):
        if type == "NEW":
            def scatterResult(input: torch.Tensor, dim: int) -> torch.Tensor:
                result = torch.zeros(*input.shape, input.shape[dim], device="cpu", dtype=torch.float64)
                input = input.unsqueeze(-1).transpose(dim, -1)
                integer_part = torch.floor(input).to(torch.int64)
                decimal_part = input - integer_part
                result.scatter_add_(dim, integer_part, 1 - decimal_part)
                result.scatter_add_(dim, torch.ceil(input).to(torch.int64), decimal_part)
                return result

            # 规则：对边缘的每个像素，对应于一个角度，例如512*512的图，上半部分就会对应2*256+512-2=1022个角度
            # 每个角度往图片中心做连线，每个角度都固定是由256个像素点加和
            # 对于线不正好穿过像素中心的情况，则实施线性插值
            #
            # 1022维的方向：上半圆从最左侧，顺时针增加至最右侧；下半圆从最右侧，顺时针增加至最左侧。
            with torch.no_grad():
                h2, w2 = (img_size[0] - 1) / 2, (img_size[1] - 1) / 2
                rangeX = torch.arange(img_size[1], device="cpu", dtype=torch.float64)
                rangeY = torch.arange(img_size[0], device="cpu", dtype=torch.float64)
                # 计算：左右边上的每个点，向中心连线，经过的每个x位置，对应的y值
                lr_mat = (torch.abs(w2 - rangeX) / w2).unsqueeze(0) * (rangeY - h2).unsqueeze(1) + h2  # 大小为(512,512)
                lr_res = scatterResult(lr_mat, 0)
                l_res = torch.cat((lr_res[:, 0:math.ceil(img_size[1] / 2)],
                                   torch.zeros((lr_res.shape[0], img_size[1] // 2, lr_res.shape[2]), device="cpu",
                                               dtype=torch.float64)),
                                  dim=1)
                r_res = torch.cat((torch.zeros((lr_res.shape[0], img_size[1] // 2, lr_res.shape[2]), device="cpu",
                                               dtype=torch.float64),
                                   lr_res[:, img_size[1] // 2:]),
                                  dim=1)
                # 计算：上下边上的每个点，向中心连线，经过的每个y位置，对应的x值
                ud_mat = (torch.abs(h2 - rangeY) / h2).unsqueeze(1) * (rangeX - w2).unsqueeze(0) + w2  # 大小为(512,512)
                ud_res = scatterResult(ud_mat, 1)
                # 拼接组合出最终结果
                h2f, h2c = img_size[0] // 2, math.ceil(img_size[0] / 2)
                vote_mask_c_up = torch.cat([l_res[:h2c, :, 1:h2f].flip([2]), ud_res[:h2c], r_res[:h2c, :, 1:h2f]],
                                           dim=2)
                vote_mask_c_down = torch.cat(
                    [r_res[h2f:, :, h2c:-1], ud_res[h2f:].flip([2]), l_res[h2f:, :, h2c:-1].flip([2])],
                    dim=2)

                vote_mask_c_up = torch.cat(
                    [vote_mask_c_up.to(torch.float32),
                     torch.zeros((h2f, *vote_mask_c_up.shape[1:]), device="cpu", dtype=torch.float32)], dim=0)
                vote_mask_c_down = torch.cat(
                    [torch.zeros((h2f, *vote_mask_c_down.shape[1:]), device="cpu", dtype=torch.float32),
                     vote_mask_c_down.to(torch.float32)], dim=0)
        else:
            raise NotImplementedError()

        return vote_mask_c_up.to(device).contiguous(), vote_mask_c_down.to(device).contiguous()

    def forward(self, pers):
        # conv1
        featmap_x = self.conv1_x(pers)
        featmap_cup = self.conv1_cup(pers)
        featmap_cdown = self.conv1_cdown(pers)
        featmap_y = self.conv1_y(pers)

        hough_x_vote = featmap_x.sum(dim=2, keepdim=True)
        hough_x_vote_reshape = hough_x_vote.reshape(hough_x_vote.shape[0], -1, self.pers_h)
        hough_y_vote = featmap_y.sum(dim=3, keepdim=True)
        hough_y_vote_reshape = hough_y_vote.reshape(hough_y_vote.shape[0], -1, self.pers_h)
        # 中心线投票
        if self.vote_mask_c_up is None:
            self.makeVoteMask(featmap_cup.shape[2:4], featmap_cup.device)

        hough_c_up_vote = torch.matmul(featmap_cup.reshape(*featmap_cup.shape[0:2], -1), self.vote_mask_c_up)
        hough_c_down_vote = torch.matmul(featmap_cdown.reshape(*featmap_cdown.shape[0:2], -1), self.vote_mask_c_down)

        # conv2: conv in hough space
        hough_feat = torch.cat(
            [self.conv2_x(hough_x_vote_reshape.unsqueeze(-1)), self.conv2_y(hough_y_vote_reshape.unsqueeze(-1))],
            dim=3)
        hough_feat_cud = torch.cat(
            [self.conv2_cup(hough_c_up_vote.unsqueeze(-1)), self.conv2_cdown(hough_c_down_vote.unsqueeze(-1))],
            dim=3)

        return hough_feat, hough_feat_cud

    def __repr__(self):
        return "FeatureShape(H={}, W={}, C={}), Perspective Length (distance_bin_num={}, fov={})".format(
            self.input_h, self.input_w, self.input_feat, self.pers_h, self.fov)


class HoughNewUpSampler(nn.Module):
    def __init__(self, upsample_rate: int):
        super().__init__()
        self.ul = nn.Upsample(scale_factor=(upsample_rate, 1), mode='bilinear', align_corners=False)
        self.um = nn.Upsample(scale_factor=(upsample_rate, 1), mode='bilinear', align_corners=False)
        self.ur = nn.Upsample(scale_factor=(upsample_rate, 1), mode='bilinear', align_corners=False)

    def forward(self, x):
        # 仅适用于偶数尺寸正方形图片的处理（因为原始图片的宽和高没有传进来，这里就默认为正方形来做上采样了）
        assert (x.shape[2] + 2) % 4 == 0, "仅适用于偶数尺寸正方形图片的处理"
        img_half_size = (x.shape[2] + 2) // 4
        l = self.ul(x[:, :, 0:img_half_size])
        m = self.um(x[:, :, img_half_size - 1:3 * img_half_size - 1])
        r = self.ur(x[:, :, 3 * img_half_size - 2:])
        return torch.cat([l[:, :, :-1], m, r[:, :, 1:]], dim=2)


class FusionHoughStage(nn.Module):
    def __init__(self, cfg, type: str, c_ori, c0, c1, c2, c3, c4, upsample_rate=None, upsampler_class=None):
        super(FusionHoughStage, self).__init__()
        self.type = type
        self.cfg = cfg
        if upsample_rate is None:
            upsample_rate = [2, 4, 8, 8, 8]

        def getSampler(u):
            if u == 1:
                return nn.Identity()
            elif upsampler_class is not None:
                return upsampler_class(u)
            else:
                return nn.Upsample(scale_factor=(u, 1), mode='bilinear', align_corners=False)

        self.upsamplers = nn.ModuleList([
            getSampler(u) for u in upsample_rate
        ])

        self.c_total = c0 + c1 + c2 + c3 + c4

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.c_total, self.c_total // 2, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(self.c_total // 2),
            nn.ReLU(),
            nn.Conv2d(self.c_total // 2, self.c_total // 2, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(self.c_total // 2),
            nn.ReLU(),
            nn.Conv2d(self.c_total // 2, 1, kernel_size=(1, 1), padding=(0, 0)),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.c_total, self.c_total // 2, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(self.c_total // 2),
            nn.ReLU(),
            nn.Conv2d(self.c_total // 2, self.c_total // 2, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(self.c_total // 2),
            nn.ReLU(),
            nn.Conv2d(self.c_total // 2, 1, kernel_size=(1, 1), padding=(0, 0)),
        )
        self.use_different_conv1 = True
        # self.conv2 = nn.Sequential(nn.Conv2d(self.c_total // 2, 1, kernel_size=(1, 1), padding=(0, 0), bias=False))

    def forward(self, x):
        concat_feat = torch.cat([sam(t) for t, sam in zip(x, self.upsamplers)], 1)
        if self.use_different_conv1:
            feat = torch.cat([self.conv1(concat_feat[:, :, :, 0:1]), self.conv1_2(concat_feat[:, :, :, 1:2])], dim=3)
        else:
            feat = self.conv1(concat_feat)
        prob = feat  # self.conv2(feat)
        # concat_feat = concat_feat.permute(0,2,1,3).reshape(f_ori.shape[0], 256, -1)
        # prob = self.linear(concat_feat).unsqueeze(1)
        return prob
