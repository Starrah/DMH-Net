import argparse
import math
import warnings
from typing import List, Optional, Tuple, Dict

import numpy as np
import py360convert
import scipy.signal
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from config import cfg_from_yaml_file, cfg, cfg_from_list
from e2plabel.e2plabelconvert import VIEW_NAME, VIEW_ARGS
from eval_cuboid import test
from eval_general import test_general
from layers import PerspectiveE2PP2E
from misc.utils import pipeload
from model import DMHNet, ENCODER_RESNET, ENCODER_DENSENET, ENCODER_HOUGH
from perspective_dataset import PerspectiveDataset
from postprocess.GDSolver import solve
from postprocess.LayoutNetv2 import LayoutNetv2PostProcessMain


def line3DConvertCore(cfg, line: List[torch.Tensor], fov, img_hw, view_idx, dis_d=None, dis_u=None, dis_f=None,
                      yline_mode="ud") -> torch.Tensor:
    """
    线表达的格式：(n,8) n是线的总数（必定等于lines[view_idx]中的各tensor大小之和），8维依次表示有效位、类型、标志、x、y、z、所在面的原始类型、所在面的序号。
    有效位为1时：
    类型：0-x值改变的线，1-y值改变的线，2-z值改变的线。
    标志：1表示作为pred结果的线，0表示一般的线 TODO
    原始类型：在视图中的线类型.。0:xleft 1:xright 2:yup 3:ydown 4:cupleft 5:cupright 6:cdownleft 7:cdownright
    坐标系规定：相机为原点，前向(F)为y轴正向，右向(R)为x轴正向，上向(U)为z轴正向。
    """
    ratio = lineCoordToRatio(cfg, line, img_hw)
    fov_hori, fov_vert = fov
    # 以视图中的内容作为坐标系，获得所有线的表达
    result = []

    # x方向线的处理
    for r, v in zip(ratio[0], line[0]):
        isXright = v >= img_hw[1] / 2
        try:
            if isXright:
                # xright
                x = dis_f * calLineAngleTan(r, fov_hori)
                result.append([1, 2, 0, x, dis_f, math.nan, 1, view_idx])
            else:
                # xleft
                x = dis_f * calLineAngleTan(r, fov_hori)
                result.append([1, 2, 0, -x, dis_f, math.nan, 0, view_idx])
        except TypeError as e:
            if len(e.args) > 0 and e.args[0].find("NoneType") != -1:
                # 传入参数为None的情况，不处理即可
                result.append([0, 2, 0, r, v, math.nan, 1 if isXright else 0, view_idx])
            else:
                raise e

    # y方向线的处理
    for r, v in zip(ratio[1], line[1]):
        isYdown = v >= img_hw[0] / 2
        try:
            if isYdown:
                if yline_mode[1] == "d":
                    y = dis_d / calLineAngleTan(r, fov_vert)
                    result.append([1, 0, 0, math.nan, y, -dis_d, 3, view_idx])
                elif yline_mode[1] == "f":
                    z = dis_f * calLineAngleTan(r, fov_vert)
                    result.append([1, 0, 0, math.nan, dis_f, -z, 3, view_idx])
                else:
                    assert False
            else:
                # yup
                if yline_mode[0] == "u":
                    y = dis_u / calLineAngleTan(r, fov_vert)
                    result.append([1, 0, 0, math.nan, y, dis_u, 2, view_idx])
                elif yline_mode[0] == "f":
                    z = dis_f * calLineAngleTan(r, fov_vert)
                    result.append([1, 0, 0, math.nan, dis_f, z, 2, view_idx])
                else:
                    assert False
        except TypeError as e:
            if len(e.args) > 0 and e.args[0].find("NoneType") != -1:
                # 传入参数为None的情况，不处理即可
                result.append([0, 0, 0, r, v, math.nan, 3 if isYdown else 2, view_idx])
            else:
                raise e

    # cup线的处理
    for r, v in zip(ratio[2], line[2]):
        isRight = r >= 0
        try:
            x = dis_u * r
            result.append([1, 1, 0, x, math.nan, dis_u, 5 if isRight else 4, view_idx])
        except TypeError as e:
            if len(e.args) > 0 and e.args[0].find("NoneType") != -1:
                # 传入参数为None的情况，不处理即可
                result.append([0, 1, 0, r, v, math.nan, 5 if isRight else 4, view_idx])
            else:
                raise e

    # cdown线的处理
    for r, v in zip(ratio[3], line[3]):
        isRight = r >= 0
        try:
            x = dis_d * r
            result.append([1, 1, 0, x, math.nan, -dis_d, 5 if isRight else 4, view_idx])
        except TypeError as e:
            if len(e.args) > 0 and e.args[0].find("NoneType") != -1:
                # 传入参数为None的情况，不处理即可
                result.append([0, 1, 0, r, v, math.nan, 5 if isRight else 4, view_idx])
            else:
                raise e

    result = torch.tensor(result, dtype=torch.float32, device=line[0].device)
    if len(result) == 0:
        result = result.new_zeros((0, 8))
    return result


def line3DConvert(cfg, line: List[torch.Tensor], fov, img_hw, view_idx, dis_d=None, dis_u=None, dis_f=None,
                  yline_mode="ud") -> torch.Tensor:
    r = line3DConvertCore(cfg, line, fov, img_hw, view_idx, dis_d=dis_d, dis_u=dis_u, dis_f=dis_f,
                          yline_mode=yline_mode)

    if VIEW_NAME[view_idx] == "F":
        pass
    elif VIEW_NAME[view_idx] == "B":
        r[:, 3:5] *= -1
    elif VIEW_NAME[view_idx] == "L":
        oriy = r[:, 4].clone()
        r[:, 4] = r[:, 3]  # y <- x
        r[:, 3] = -oriy  # x <- -y

        yLineMask = r[:, 1] == 1  # 交换线类别中的x线与y线
        r[r[:, 1] == 0, 1] = 1
        r[yLineMask, 1] = 0
    elif VIEW_NAME[view_idx] == "R":
        oriy = r[:, 4].clone()
        r[:, 4] = -r[:, 3]  # y <- -x
        r[:, 3] = oriy  # x <- y

        yLineMask = r[:, 1] == 1  # 交换线类别中的x线与y线
        r[r[:, 1] == 0, 1] = 1
        r[yLineMask, 1] = 0
    elif VIEW_NAME[view_idx] == "D":
        oriy = r[:, 4].clone()
        r[:, 4] = r[:, 5]  # y <- z
        r[:, 5] = -oriy  # z <- -y

        yLineMask = r[:, 1] == 1  # 交换线类别中的z线与y线
        r[r[:, 1] == 2, 1] = 1
        r[yLineMask, 1] = 2
    elif VIEW_NAME[view_idx] == "U":
        oriy = r[:, 4].clone()
        r[:, 4] = -r[:, 5]  # y <- -z
        r[:, 5] = oriy  # z <- y

        yLineMask = r[:, 1] == 1  # 交换线类别中的z线与y线
        r[r[:, 1] == 2, 1] = 1
        r[yLineMask, 1] = 2
    else:
        assert False
    return r


def allLinesConvert(cfg, lines: List[List[torch.Tensor]], img_hw, camera_height, dis_u, dis_box, view_args,
                    extra: Optional[List[List[torch.Tensor]]] = None):
    line_result = []
    # 四个中间面
    for view_idx in range(4):
        r = line3DConvert(cfg, lines[view_idx], view_args[view_idx][0], img_hw, view_idx, dis_d=camera_height,
                          dis_u=dis_u, dis_f=dis_box[view_idx] if dis_box is not None else None, yline_mode="ud")
        line_result.append(r)
    # 上面
    view_idx = 4
    r = line3DConvert(cfg, lines[view_idx], view_args[view_idx][0], img_hw, view_idx, dis_f=dis_u,
                      dis_u=dis_box[2] if dis_box is not None else None,
                      dis_d=dis_box[0] if dis_box is not None else None, yline_mode="ff")
    line_result.append(r)
    # 下面
    view_idx = 5
    r = line3DConvert(cfg, lines[view_idx], view_args[view_idx][0], img_hw, view_idx, dis_f=camera_height,
                      dis_u=dis_box[0] if dis_box is not None else None,
                      dis_d=dis_box[2] if dis_box is not None else None, yline_mode="ff")
    line_result.append(r)
    line_result = torch.cat(line_result, 0)

    if extra is not None:  # 附加extra信息
        line_result = torch.cat([line_result, torch.cat([torch.cat(a) for a in extra]).unsqueeze(1)], 1)
    return line_result


def classifyLine(line) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    把线条分成12类：三个值分别代表xyz与0的关系，-表示线沿该方向。
    0->> 1->< 2-<> 3-<< 4>-> 5>-< 6<-> 7<-< 8>>- 9><- 10<>- 11<<-
    类别信息附加在线的索引最后一维处
    :param line (n, k)
    :return Tuple[(n, k+1)-处理后的附加了类别信息的线结构，前k维是输入的原始线表示，最后一维是类别；
     按照12类别分类后的线的数组]
    """
    classify_arr = []
    for l in line:
        if l[1] == 0:
            c = (l[4] < 0) * 2 + (l[5] < 0) + 0
        elif l[1] == 1:
            c = (l[3] < 0) * 2 + (l[5] < 0) + 4
        elif l[1] == 2:
            c = (l[3] < 0) * 2 + (l[4] < 0) + 8
        classify_arr.append(c)
    line = torch.cat([line, line.new_tensor(classify_arr).unsqueeze(1)], 1)
    classified_result = []
    for i in range(12):
        classified_result.append(line[line[:, -1] == i])

    return line, classified_result


def lwhToPeaks(cfg, lwh: torch.Tensor, img_hw) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    :return 6个面、4种线、两个数第一个代表霍夫域值、第二个代表线的可见长度百分比
    """
    lwh = torch.abs(lwh)
    result = [[[] for _ in range(4)] for _ in range(6)]
    TABLE = [
        [3, [0, 1, 5, 4]],
        [1, [3, 2, 5, 4]],
        [2, [1, 0, 5, 4]],
        [0, [2, 3, 5, 4]],
        [5, [0, 1, 2, 3]],
        [4, [0, 1, 3, 2]]
    ]

    for view_idx in range(6):
        ratio = 1 - (lwh[TABLE[view_idx][1]] / lwh[TABLE[view_idx][0]])
        with torch.set_grad_enabled(True):
            length = ratio.new_ones(4)
        pointPlace = [(img_hw[1] - 1) / 2 * ratio[0], (img_hw[1] - 1) / 2 * (2 - ratio[1]),
                      (img_hw[0] - 1) / 2 * ratio[2], (img_hw[0] - 1) / 2 * (2 - ratio[3])]

        # 找交点、refine 水平竖直线长度、计算中心线
        for i in range(2):
            for j in range(2, 4):
                if not (ratio[i] > 0 and ratio[j] > 0):
                    continue
                with torch.set_grad_enabled(True):
                    length[i] = length[i] - (ratio[j] / 2)
                    length[j] = length[j] - (ratio[i] / 2)
                    lenCLine = min(ratio[i], ratio[j]).clone()
                houghParamCLine, isCDown = PerspectiveDataset.coord2AngleValue(pointPlace[i], pointPlace[j], img_hw)

                result[view_idx][3 if isCDown == 1 else 2].append((houghParamCLine, lenCLine))

        # 添加水平和竖直的peak
        for i in range(4):
            if ratio[i] > 0:
                result[view_idx][0 if i < 2 else 1].append((pointPlace[i], length[i]))

        for t in range(4):
            l = result[view_idx][t]
            l2 = (torch.stack([tup[0] for tup in l]) if len(l) > 0 else lwh.new_zeros(0),
                  torch.stack([tup[1] for tup in l]) if len(l) > 0 else lwh.new_zeros(0))
            result[view_idx][t] = l2

    return result


def solveActualHeightByIOU(d: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    :param d shape(n, 4)，四维分别代表(y+, x+, y-, x-)。u格式相同。
    :return shape(n) 表示相机到上方实际高度，与当前假设高度的比值
    """
    changeToNegative = d.new_tensor([-1, 1, -1, 1])
    d, u = (d[[3, 1, 2, 0]] * changeToNegative).unsqueeze(0), (u[[3, 1, 2, 0]] * changeToNegative).unsqueeze(0)

    class UpperIOUModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(d.shape[0], requires_grad=True))

        def forward(self):
            return (1 - calculateIOU(d, u * self.w).diagonal()).mean()

    module = UpperIOUModule().to(d.device)
    module = solve(module, max_iter=100, lr=1e-2, stop_tol=1e-3, stop_range=5)
    return module.w.data


def interpolate(vector: torch.Tensor, position: torch.Tensor):
    floored = torch.floor(position).to(torch.int64)
    flooredAdd1 = floored + 1
    floored = torch.clamp(floored, 0, len(vector) - 1)
    flooredAdd1 = torch.clamp(flooredAdd1, 0, len(vector) - 1)
    remain = position - floored
    return remain * vector[flooredAdd1] + (1 - remain) * vector[floored]


def lossFunction(output, img_idx, peakss):
    preds_xy = torch.sigmoid(output["p_preds_xy"][img_idx]).to(peakss[0][0][0].device)
    preds_cud = torch.sigmoid(output["p_preds_cud"][img_idx]).to(peakss[0][0][0].device)

    views_result = []
    for view_idx, ((xPeaks, xLength), (yPeaks, yLength), (cupPeaks, cupLength),
                   (cDownPeaks, cDownLength)) in enumerate(peakss):
        fourPreds = [preds_xy[view_idx, :, 0], preds_xy[view_idx, :, 1],
                     preds_cud[view_idx, :, 0], preds_cud[view_idx, :, 1]]
        fourPeaks = [xPeaks, yPeaks, cupPeaks, cDownPeaks]
        fourLength = [xLength, yLength, cupLength, cDownLength]
        view_result = []
        for preds, peaks, length in zip(fourPreds, fourPeaks, fourLength):
            scores = interpolate(preds, peaks)
            scores = scores * length
            view_result.append(scores)
        view_result = torch.cat(view_result)
        views_result.append(view_result)
    views_result = torch.cat(views_result)
    final_result = views_result.sum()
    final_result.requires_grad_(True)
    return 20 - final_result


def solveLwh(cfg, output, img_idx, start_lwh, img_hw):
    class LWHSolveModule(nn.Module):
        def __init__(self):
            super().__init__()
            startParam = start_lwh[[0, 1, 2, 3, 5]]
            self.param = nn.Parameter(startParam.clone().detach().requires_grad_(True), requires_grad=True)

        def lwh(self):
            return torch.cat([self.param[0:4], self.param.new_tensor([-1.6]), self.param[4:5]])

        def forward(self):
            lwh = self.lwh()
            peaks = lwhToPeaks(cfg, lwh, img_hw)
            return lossFunction(output, img_idx, peaks)

    module = LWHSolveModule().to("cpu")
    module = solve(module, max_iter=100, lr=1e-2, stop_tol=1e-3, stop_range=5)
    return module.lwh().data, torch.abs(module.forward())


def linesGTProcess_PretendDisUThenOptimIOU(cfg, lines: List[List[torch.Tensor]], img_hw, camera_height=1.6,
                                           view_args=VIEW_ARGS,
                                           preset_dis_u=1.6):
    """
    假设一个相机到天花板的高度，完整的估计出上方的方框的尺寸，再通过优化IOU问题求高度
    """
    with torch.no_grad():
        line_result = allLinesConvert(cfg, lines, img_hw, camera_height, preset_dis_u, None, view_args)
        line_result, classified = classifyLine(line_result)
        # 注意，到此时，z线（竖直线）的坐标还没有算出
        # 断言：对于GT，每个类别中的、在原始perspect中是xyline的线应当有且仅有一条
        # 除了没有算出的z线外，余下的八条线就构成了下框和上框
        # 优化两者的IOU最大问题，求出了height。利用height
        dis_u = preset_dis_u
        downbox, upbox, _ = calculateUDBox(cfg, classified, useCLine=False, useZLine=False,
                                           require_onlyone=cfg.DATA.TYPE == "cuboid")
        w = solveActualHeightByIOU(downbox, upbox).item()
        dis_u = dis_u * w
        dis_box = (downbox + upbox * w) / 2
        # 单独重算一次line_result，以让z线能产生结果
        line_result = allLinesConvert(cfg, lines, img_hw, camera_height, dis_u, dis_box, view_args)
        return line_result, dis_box.new_tensor(
            [-dis_box[3], dis_box[1], -dis_box[2], dis_box[0], -camera_height, dis_u])


def updateProb(line: torch.Tensor, amount: float, mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
    toAdd = torch.zeros_like(line)
    if mask is not None:
        toAdd[mask, 8] = amount
    else:
        toAdd[:, 8] = amount
    return line + toAdd


def calculateUDBox(cfg, classified: List[torch.Tensor], useCLine=True, useZLine=True, c_line_prob_punish=None,
                   z_line_prob_punish=None, require_onlyone=False) -> Tuple[
    torch.Tensor, torch.Tensor, Optional[dict]]:
    """
    根据已经算好的、分类过的三维线结果，重算上下框。
    重算上下框的策略：每个框对应三组线：两组竖直线、一组本身对应的天花板线或地板线。
    给竖直线的概率均减去一个常数，再把三组线merge到一块、取概率最大者。
    :return Tuple[downbox, upbox, extra] box的格式为(y+, x+, y-, x-),extra是dict或None
    """
    c_line_prob_punish = c_line_prob_punish if c_line_prob_punish is not None else cfg.POST_PROCESS.CPP
    z_line_prob_punish = z_line_prob_punish if z_line_prob_punish is not None else cfg.POST_PROCESS.ZPP

    z_class_seq = [[8, 10], [8, 9], [9, 11], [10, 11]] if useZLine else [[]] * 4
    result = []
    extra = None

    for mainClass, zClasses, name in zip([1, 5, 3, 7, 0, 4, 2, 6], z_class_seq * 2,
                                         ["_↑", "_→", "_↓", "_←", "¯↑", "¯→", "¯↓", "¯←"]):
        if not useZLine:
            t = classified[mainClass]
            notCLine_mask = torch.logical_and(0 <= t[:, 6], t[:, 6] <= 3)
            valid_mask = t[:, 0] == 1 if useCLine else torch.logical_and(t[:, 0] == 1, notCLine_mask)
            t = t[valid_mask]
            if useCLine and c_line_prob_punish is not None:  # cline惩罚
                t = updateProb(t, -c_line_prob_punish, ~notCLine_mask)  # 减去被惩罚的概率值
            r = t
        else:
            r = []
            t = classified[mainClass]
            notCLine_mask = torch.logical_and(0 <= t[:, 6], t[:, 6] <= 3)
            valid_mask = t[:, 0] == 1 if useCLine else torch.logical_and(t[:, 0] == 1, notCLine_mask)
            t = t[valid_mask]
            if useCLine and c_line_prob_punish is not None:  # cline惩罚
                t = updateProb(t, -c_line_prob_punish, ~notCLine_mask)  # 减去被惩罚的概率值
            r.append(t)
            for zClass in zClasses:
                t = classified[zClass]
                notCLine_mask = torch.logical_and(0 <= t[:, 6], t[:, 6] <= 3)
                valid_mask = t[:, 0] == 1 if useCLine else torch.logical_and(t[:, 0] == 1, notCLine_mask)
                t = t[valid_mask]
                if useCLine and c_line_prob_punish is not None:  # cline惩罚
                    t = updateProb(t, -c_line_prob_punish, ~notCLine_mask)  # 减去被惩罚的概率值
                if z_line_prob_punish is not None:  # zline惩罚（对zCLass的所有类适用）
                    t = updateProb(t, -z_line_prob_punish)  # 减去被惩罚的概率值
                r.append(t)
            r = torch.cat(r, 0)

        pickWhichAxis = (1 if mainClass <= 3 else 0) + 3
        if require_onlyone:
            if len(r) != 1:
                warnings.warn(
                    "calculateUDBox assertion require_onlyone fail! for mainClass {:d}, has {:} lines".format(mainClass,
                                                                                                              len(r)))
            else:
                r[:, 2] = 1
            result.append(r[:, pickWhichAxis].mean().abs())

        else:
            if len(r) > 0:
                idx = r[:, 8].argmax()
                r[idx, 2] = 1
                result.append(r[idx, pickWhichAxis].abs())
            else:
                warnings.warn("calculateUDBox: no line for mainClass {:s}({:d})!".format(name, mainClass))
                result.append(cfg.POST_PROCESS.DEFAULT_DISTANCE)
                if extra is None: extra = {}
                if "noline" not in extra: extra["noline"] = []
                extra["noline"].append(name)

    if extra is not None and "noline" in extra: extra["noline"] = " ".join(extra["noline"])
    return classified[0].new_tensor(result[0:4]), classified[0].new_tensor(result[4:8]), extra


def linesPredProcess_PretendDisUThenOptimIOU(cfg, lines: List[List[torch.Tensor]], probs: List[List[torch.Tensor]],
                                             img_hw, camera_height=1.6,
                                             view_args=VIEW_ARGS,
                                             preset_dis_u=1.6):
    """
    假设一个相机到天花板的高度，完整的估计出上方的方框的尺寸，再通过优化IOU问题求高度
    """
    with torch.no_grad():
        line_result = allLinesConvert(cfg, lines, img_hw, camera_height, preset_dis_u, None, view_args, extra=probs)
        line_result, classified = classifyLine(line_result)

        # 注意，到此时，z线（竖直线）的坐标还没有算出
        # 对于Pred，除了没有算出的z线外，余下的8类中每类取概率最大的线，这八条线就构成了下框和上框
        # 优化两者的IOU最大问题，求出了height，同时也就求出了长宽高
        dis_u = preset_dis_u
        downbox, upbox, extra = calculateUDBox(cfg, classified, useZLine=False)
        w = solveActualHeightByIOU(downbox, upbox).item()
        dis_u = dis_u * w
        dis_box = (downbox + upbox * w) / 2

        extra_nz = extra["noline"] if extra is not None and "noline" in extra else None

        for i in range(cfg.POST_PROCESS.ITER if extra_nz is None else cfg.POST_PROCESS.ITER_NZ):
            # 输入：到房间顶的距离、下框、上框
            # 过程：1.根据下框和上框求解距离因子，更新到房间顶的距离；
            # 2.根据新的到房间顶的距离，和上框下框（上框要用刚算出的w缩放一下）的平均值，重求解空间中所有线；
            # 3.根据线的完全结果（包含了竖直线和框线的结果）更新下框和上框
            # 根据此规则迭代，道理上就可以完成对框线的优化
            olddownbox, oldupbox = downbox, upbox
            line_result = allLinesConvert(cfg, lines, img_hw, camera_height, dis_u, dis_box, view_args, extra=probs)
            line_result, classified = classifyLine(line_result)
            downbox, upbox, extra = calculateUDBox(cfg, classified)
            w = solveActualHeightByIOU(downbox, upbox).item()
            dis_u = dis_u * w
            dis_box = (downbox + upbox * w) / 2
            if torch.mean(torch.abs(downbox - olddownbox)) < 1e-3 and torch.mean(torch.abs(upbox - oldupbox)) < 1e-3:
                break

        line_result = allLinesConvert(cfg, lines, img_hw, camera_height, dis_u, dis_box, view_args, extra=probs)

        if extra_nz is not None:
            if extra is None: extra = {}
            extra["nz"] = extra_nz

        return line_result, dis_box.new_tensor(
            [-dis_box[3], dis_box[1], -dis_box[2], dis_box[0], -camera_height, dis_u]), extra


def findPeaks(cfg, vector: torch.Tensor) -> torch.Tensor:
    # TODO 调参等
    # locs = scipy.signal.find_peaks_cwt(vector.cpu().numpy(), np.arange(10, 60))
    # locs, _ = scipy.signal.find_peaks(vector.cpu().numpy(), height=0.5, distance=60, prominence=0.4)
    locs, _ = scipy.signal.find_peaks(vector.cpu().numpy(), distance=cfg.POST_PROCESS.get("PEAK_DISTANCE", 60),
                                      height=cfg.POST_PROCESS.PEAK_HEIGHT,
                                      prominence=cfg.POST_PROCESS.PEAK_PROMINENCE)
    return vector.new_tensor(locs)


def predProbMap_PretendDisUThenOptimIOU(cfg, output, img_idx, img_hw, camera_height=1.6, view_args=VIEW_ARGS,
                                        preset_dis_u=1.6):
    with torch.no_grad():
        lines, probs = extractPredPeaks(cfg, output, img_idx)
        return linesPredProcess_PretendDisUThenOptimIOU(cfg, lines, probs, img_hw, camera_height, view_args,
                                                        preset_dis_u)


def extractPredPeaks(cfg, output, img_idx):
    lines, probs = [], []
    for view_idx in range(6):
        view_line, view_prob = [], []
        for signal in [output["p_preds_xy"][img_idx, view_idx, :, 0],
                       output["p_preds_xy"][img_idx, view_idx, :, 1],
                       output["p_preds_cud"][img_idx, view_idx, :, 0],
                       output["p_preds_cud"][img_idx, view_idx, :, 1]]:
            signal = torch.sigmoid(signal)
            # 前半部分和后半部分分别寻找peak
            mid = len(signal) // 2
            peak = torch.cat([findPeaks(cfg, signal[0:mid]), mid + findPeaks(cfg, signal[mid:])])
            prob = signal[peak.to(torch.int64)]
            view_line.append(peak)
            view_prob.append(prob)
        lines.append(view_line)
        probs.append(view_prob)
    return lines, probs


def findPeaks2D8Points(cfg, matrix: torch.Tensor) -> torch.Tensor:
    # 每一列，从上、下半部分各取出值最大的n个点， 取平均
    h2f = matrix.shape[0] // 2
    column_vec = torch.cat([matrix[0:h2f].topk(cfg.POST_PROCESS.EMASK.ROW_CHOOSE_N, dim=0)[0],
                            matrix[h2f:].topk(cfg.POST_PROCESS.EMASK.ROW_CHOOSE_N, dim=0)[0]], 0).mean(0)
    result = []
    columns = []
    # 对这个东西进行逐段的峰值提取
    for i in range(4):
        begin, end = matrix.shape[1] * i // 4, matrix.shape[1] * (i + 1) // 4
        seq = column_vec[begin:end]
        peaks1D = findPeaks(cfg, seq).to(torch.int64)
        if len(peaks1D) > 0:
            best_peak = peaks1D[seq[peaks1D].argmax()].item() + begin
        else:
            warnings.warn("when find 2d peaks for emask, len(peaks1D) == 0 when calculating column!")
            best_peak = (begin + end) // 2
        columns.append(best_peak)
    for column in columns:
        for i in reversed(range(2)):
            begin, end = matrix.shape[0] * i // 2, matrix.shape[0] * (i + 1) // 2
            seq = matrix[begin:end, column]
            peaks1D = findPeaks(cfg, seq).to(torch.int64)
            if len(peaks1D) > 0:
                best_peak = peaks1D[seq[peaks1D].argmax()].item() + begin
            else:
                warnings.warn("when find 2d peaks for emask, len(peaks1D) == 0 when calculating column!")
                best_peak = (begin + end) // 2
            result.extend([(column, best_peak)])
    return matrix.new_tensor(result)


def calPredCorIdByEMask(cfg, emask_img, z0=50):
    type_weight = emask_img.new_tensor(cfg.POST_PROCESS.EMASK.TYPE_WEIGHT)
    type_weight /= type_weight.sum()
    emask_score = (type_weight.unsqueeze(-1).unsqueeze(-1) * emask_img).sum(dim=0)

    coords = findPeaks2D8Points(cfg, emask_score).cpu().numpy()
    indices = np.repeat(np.argsort(coords[1::2, 0]) * 2, 2)
    indices[0::2] += 1
    coords = coords[indices]

    xyz = py360convert.uv2unitxyz(py360convert.coor2uv(coords, *emask_img.shape[1:3]))
    z1 = (xyz[1::2, 1] / xyz[0::2, 1]).mean() * z0

    return coords, z0, z1.item()


def calMetrics_PretendDisUThenOptimIOU(cfg, input, output, img_idx, optimization, camera_height=1.6,
                                       view_args=VIEW_ARGS, preset_dis_u=1.6) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[
        str, torch.Tensor]]:
    with torch.no_grad():
        img_hw = input["p_imgs"].shape[-2:]
        e_img_hw = input["e_img"].shape[-2:]
        gt_lines, gt_lwh = linesGTProcess_PretendDisUThenOptimIOU(cfg, input["peaks"][img_idx], img_hw, camera_height,
                                                                  view_args, preset_dis_u)
        gt_cor_id = input["cor"][img_idx]

        pred_lines, pred_lwh, pred_extra = predProbMap_PretendDisUThenOptimIOU(cfg, output, img_idx, img_hw,
                                                                               camera_height,
                                                                               view_args, preset_dis_u)
        if optimization:
            pred_lwh, _ = solveLwh(cfg, output, img_idx, pred_lwh, img_hw)
            pred_peaks, pred_probs = extractPredPeaks(cfg, output, img_idx)
            pred_lines = allLinesConvert(cfg, pred_peaks, img_hw, -pred_lwh[4], pred_lwh[5], pred_lwh[[3, 1, 2, 0]],
                                         view_args, pred_probs)

        corner_method = cfg.POST_PROCESS.get("CORNER_METHOD", "lwh")
        if corner_method == "lwh":
            pred_cor_id_np, z0, z1 = cvtPredLwhToEquirecCornerCoords(pred_lwh, e_img_hw)
        elif corner_method == "emask":
            pred_cor_id_np, z0, z1 = calPredCorIdByEMask(cfg, output["p_preds_emask"][img_idx])
        pred_cor_id = pred_lwh.new_tensor(pred_cor_id_np)
        # 统一到GT所在的device上进行计算
        pred_lines, pred_lwh = pred_lines.to(gt_lines.device), pred_lwh.to(gt_lwh.device)
        metrics = {}

        # 调用HorizonNet的算指标代码进行计算
        t = {}
        if cfg.POST_PROCESS.get("TEST_WITH_BOTH", False):
            test_general(pred_cor_id_np, gt_cor_id.cpu().numpy(), e_img_hw[1], e_img_hw[0], t)
            t["3DIoU-general"] = t["3DIoU"]
            test(pred_cor_id_np, z0, z1, gt_cor_id.cpu().numpy(), e_img_hw[1], e_img_hw[0], t)
        elif cfg.DATA.TYPE == "cuboid":
            test(pred_cor_id_np, z0, z1, gt_cor_id.cpu().numpy(), e_img_hw[1], e_img_hw[0], t)
        elif cfg.DATA.TYPE == "general":
            test_general(pred_cor_id_np, gt_cor_id.cpu().numpy(), e_img_hw[1], e_img_hw[0], t)
        else:
            assert False

        for k in t:
            metrics[k] = pred_lwh.new_tensor(t[k]) if not isinstance(t[k], str) else t[k]

        if pred_extra is not None:
            metrics.update(pred_extra)

        return (gt_lines, gt_lwh, gt_cor_id), (pred_lines, pred_lwh, pred_cor_id), metrics


def calMetrics_PretendDisUThenOptimIOUForV2(cfg, input, output, img_idx, optimization, camera_height=1.6,
                                            view_args=VIEW_ARGS, preset_dis_u=1.6) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[
        str, torch.Tensor], torch.Tensor]:
    with torch.no_grad():
        img_hw = input["p_imgs"].shape[-2:]
        e_img_hw = input["e_img"].shape[-2:]
        gt_lines, gt_lwh = linesGTProcess_PretendDisUThenOptimIOU(cfg, input["peaks"][img_idx], img_hw, camera_height,
                                                                  view_args, preset_dis_u)
        gt_cor_id = input["cor"][img_idx]

        pred_lines, pred_lwh, pred_extra = predProbMap_PretendDisUThenOptimIOU(cfg, output, img_idx, img_hw,
                                                                               camera_height,
                                                                               view_args, preset_dis_u)
        if optimization:
            pred_lwh, err_score = solveLwh(cfg, output, img_idx, pred_lwh, img_hw)
            pred_peaks, pred_probs = extractPredPeaks(cfg, output, img_idx)
            pred_lines = allLinesConvert(cfg, pred_peaks, img_hw, -pred_lwh[4], pred_lwh[5], pred_lwh[[3, 1, 2, 0]],
                                         view_args, pred_probs)

        corner_method = cfg.POST_PROCESS.get("CORNER_METHOD", "lwh")
        if corner_method == "lwh":
            pred_cor_id_np, z0, z1 = cvtPredLwhToEquirecCornerCoords(pred_lwh, e_img_hw)
        elif corner_method == "emask":
            pred_cor_id_np, z0, z1 = calPredCorIdByEMask(cfg, output["p_preds_emask"][img_idx])
        pred_cor_id = pred_lwh.new_tensor(pred_cor_id_np)
        # 统一到GT所在的device上进行计算
        pred_lines, pred_lwh = pred_lines.to(gt_lines.device), pred_lwh.to(gt_lwh.device)
        # iou3d = calculateIOU(pred_lwh.unsqueeze(0), gt_lwh.unsqueeze(0))[0, 0]
        # iou2d = calculateIOU(pred_lwh[:4].unsqueeze(0), gt_lwh[:4].unsqueeze(0))[0, 0]
        # metrics = {
        #     "box_iou3d": iou3d,
        #     "box_iou2d": iou2d
        # }
        metrics = {}

        # 调用HorizonNet的算指标代码进行计算
        t = {}
        if cfg.POST_PROCESS.get("TEST_WITH_BOTH", False):
            test_general(pred_cor_id_np, gt_cor_id.cpu().numpy(), e_img_hw[1], e_img_hw[0], t)
            t["3DIoU-general"] = t["3DIoU"]
            test(pred_cor_id_np, z0, z1, gt_cor_id.cpu().numpy(), e_img_hw[1], e_img_hw[0], t)
        elif cfg.DATA.TYPE == "cuboid":
            test(pred_cor_id_np, z0, z1, gt_cor_id.cpu().numpy(), e_img_hw[1], e_img_hw[0], t)
        elif cfg.DATA.TYPE == "general":
            test_general(pred_cor_id_np, gt_cor_id.cpu().numpy(), e_img_hw[1], e_img_hw[0], t)
        else:
            assert False

        for k in t:
            metrics[k] = pred_lwh.new_tensor(t[k]) if not isinstance(t[k], str) else t[k]

        if pred_extra is not None:
            metrics.update(pred_extra)

        return (gt_lines, gt_lwh, gt_cor_id), (pred_lines, pred_lwh, pred_cor_id), metrics, err_score


def LayoutNetv2PostProcessWrapper(cfg, input, output, img_idx) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[
        str, torch.Tensor]]:
    """
    :return Tensor<n,2>角点坐标，可以直接喂给test_general函数的那种
    """
    e_img_hw = input["e_img"].shape[-2:]
    gt_cor_id = input["cor"][img_idx]
    generatePred2DMask(cfg, input, output, img_idx)
    pred_cor_id_np = LayoutNetv2PostProcessMain(output["p_preds_emask"][img_idx].mean(0).cpu().numpy(),
                                                output["p_preds_emask"][img_idx].permute(1, 2, 0).cpu().numpy())
    pred_cor_id = input["cor"][img_idx].new_tensor(pred_cor_id_np)
    metrics = {}

    # 调用HorizonNet的算指标代码进行计算
    t = {}
    if cfg.DATA.TYPE == "cuboid":
        assert False
        test(pred_cor_id_np, z0, z1, gt_cor_id.cpu().numpy(), e_img_hw[1], e_img_hw[0], t)
    elif cfg.DATA.TYPE == "general":
        test_general(pred_cor_id_np, gt_cor_id.cpu().numpy(), e_img_hw[1], e_img_hw[0], t)
    else:
        assert False

    for k in t:
        metrics[k] = output["p_preds_emask"][img_idx].new_tensor(t[k]) if not isinstance(t[k], str) else t[k]

    return (None, None, gt_cor_id), (None, None, pred_cor_id), metrics


def postProcess(cfg, input, output, img_idx, is_valid_mode=False, camera_height=1.6, view_args=VIEW_ARGS,
                preset_dis_u=1.6) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[
        str, torch.Tensor]]:
    """
    :return (gt_lines, gt_lwh, gt_cor_id), (pred_lines, pred_lwh, pred_cor_id), metrics
    """
    method = cfg.POST_PROCESS.METHOD if not (is_valid_mode and ("METHOD_WHEN_VALID" in cfg.POST_PROCESS)) \
        else cfg.POST_PROCESS.METHOD_WHEN_VALID
    if method == "None" or method is None:
        return (None, None, input["cor"][img_idx]), (None, None, None), {}
    elif method == "geometry" or method == "optimization":
        return calMetrics_PretendDisUThenOptimIOU(cfg, input, output, img_idx, method == "optimization", camera_height,
                                                  view_args, preset_dis_u)
    elif method == "LayoutNetv2":
        return LayoutNetv2PostProcessWrapper(cfg, input, output, img_idx)
    elif method == "develop" or method == "noncuboid":
        from postprocess.noncuboid import nonCuboidPostProcess
        return nonCuboidPostProcess(cfg, input, output, img_idx)
    else:
        assert False, "不支持的POST_PROCESS.METHOD"


def calculateIOU(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    box格式：(x-,x+,y-,y+,z-,z+)。z-、z+可以不提供。
    要求同一轴上的元素，小的必须在大的的前面，否则无法算出正确结果！
    """
    assert torch.all(boxes1[:, ::2] <= boxes1[:, 1::2]) and torch.all(boxes2[:, ::2] <= boxes2[:, 1::2])
    boxes1 = boxes1.unsqueeze(1)
    boxes2 = boxes2.unsqueeze(0)
    boxes1_volume = torch.clamp(boxes1[:, :, 1::2] - boxes1[:, :, 0::2], min=0).prod(dim=2)
    boxes2_volume = torch.clamp(boxes2[:, :, 1::2] - boxes2[:, :, 0::2], min=0).prod(dim=2)

    boxes1 = boxes1.repeat(1, len(boxes2), 1)
    boxes2 = boxes2.repeat(len(boxes1), 1, 1)
    mixed = torch.stack([boxes1, boxes2], dim=3)
    distance = mixed[:, :, 1::2].min(dim=3)[0] - mixed[:, :, 0::2].max(dim=3)[0]
    intersect = torch.clamp(distance, min=0).prod(dim=2)

    iou = intersect / (boxes1_volume + boxes2_volume - intersect)
    return iou


def calLineAngleTan(x, fov: int):
    tan = np.tan(np.deg2rad(fov / 2)) if fov != 90 else 1.0
    return (1 - x) * tan


def lineCoordToRatio(cfg, line: List[torch.Tensor], img_hw):
    result = [t.clone() for t in line]
    # 10.12由于和linesPostProcess相同的原因，x、y线处理加上了-1。中心线原来算的就已经是对的了，不需要再处理了。
    # y不变线处理
    yline = result[1]
    yDownMask = yline >= (img_hw[0] - 1) / 2
    yline[yDownMask] = (img_hw[0] - 1) - yline[yDownMask]
    result[1] = yline / ((img_hw[0] - 1) / 2)

    # x不变线处理
    xline = result[0]
    xRightMask = xline >= (img_hw[1] - 1) / 2
    xline[xRightMask] = (img_hw[1] - 1) - xline[xRightMask]
    result[0] = xline / ((img_hw[1] - 1) / 2)

    # 角度线处理：
    if cfg.MODEL.HOUGH.CLINE_TYPE == "NEW":
        # 返回的是与铅垂线夹角的tan值。这个值直接乘以已知的相机高度/到天花板的距离，所得结果直接就是x坐标了。
        h2, w2 = (img_hw[0] - 1) / 2, (img_hw[1] - 1) / 2
        h2f = img_hw[0] // 2

        # cup线
        cupline = result[2]
        mask1 = cupline < h2f - 1
        mask3 = cupline > h2f + img_hw[1] - 2
        mask2 = torch.logical_not(torch.logical_or(mask1, mask3))
        cupline[mask1] = w2 / (((h2f - 1) - cupline[mask1]) - h2)
        cupline[mask2] = ((cupline[mask2] - (h2f - 1)) - w2) / h2
        cupline[mask3] = -w2 / ((cupline[mask3] - (h2f + img_hw[1] - 2)) - h2)
        result[2] = cupline

        # cdown线
        cdownline = result[3]
        mask1 = cdownline < h2f - 1
        mask3 = cdownline > h2f + img_hw[1] - 2
        mask2 = torch.logical_not(torch.logical_or(mask1, mask3))
        cdownline[mask1] = w2 / (h2 - ((h2f - 1) - cdownline[mask1]))
        cdownline[mask2] = (w2 - (cdownline[mask2] - (h2f - 1))) / h2
        cdownline[mask3] = -w2 / (h2 - (cdownline[mask3] - (h2f + img_hw[1] - 2)))
        result[3] = cdownline

    else:
        raise NotImplementedError()

    return result


def gtVisualize(cfg, lines: List[List[torch.Tensor]], img_hw, camera_height=1.6):
    """
    Open3D自带的坐标轴中，红色是x轴，绿色是y轴，蓝色是z轴！
    显示的线，红色表示x不变线，绿色表示y不变线，蓝色表示过中心的线！
    """
    lines_results, layout_param = linesGTProcess_PretendDisUThenOptimIOU(cfg, lines, img_hw)
    print(layout_param)
    from visualization import o3dRunVis, o3dDrawLines, o3dInitVis
    vis = o3dInitVis()
    o3dDrawLines(vis, lines_results, layout_param)
    o3dRunVis(vis)


def cvtPredLwhToEquirecCornerCoords(lwh, e_img_hw, z0=50):
    """
    将估计出的房间lwh格式的数据，转化为HorizonNet的eval_cuboid.py中的test函数兼容的格式。
    :param lwh <6> -x, x, -y, y, -z, z
    :return dt_cor_id, z0, z1
    """
    z1 = lwh[4] / lwh[5] * z0

    # 构造8个uv坐标
    def lwhToUv(xyz):
        u = torch.atan2(xyz[0], xyz[1])
        v = torch.atan(xyz[2] / torch.norm(xyz[0:2]))
        return torch.stack((u, v))

    uvs = []
    for i in range(8):
        uv = lwhToUv(lwh[[(i // 4) % 2 + 0, (i // 2) % 2 + 2, (i // 1) % 2 + 4]].cpu())
        uvs.append(uv)
    uvs = torch.stack(uvs)

    coords = py360convert.uv2coor(uvs.cpu().numpy(), *e_img_hw)
    indices = np.repeat(np.argsort(coords[1::2, 0]) * 2, 2)
    indices[0::2] += 1
    coords = coords[indices]

    return coords, z0, z1.item()


vote_mask_c_up_down = None


def get_vote_mask_c_up_down(cfg, p_img):
    global vote_mask_c_up_down
    if vote_mask_c_up_down is None:
        u, d = PerspectiveE2PP2E.makeVoteMaskStatic(cfg.MODEL.HOUGH.CLINE_TYPE, p_img.shape[-2:], p_img.device)
        vote_mask_c_up_down = torch.cat([u, d], -1)
    return vote_mask_c_up_down


def _cal_p_pred_2d_mask(cfg, input, img_idx, p_pred_xy_oneimage, p_pred_cud_oneimage):
    p_imgs = input["p_imgs"][img_idx]

    # 生成六个面的mask
    result_2dmask = []
    for view_idx, p_img in enumerate(p_imgs):
        # xLine的mask
        prob = p_pred_xy_oneimage[view_idx, :, 0]
        x_mat = prob.unsqueeze(0).expand(p_img.shape[1], -1)
        # yLine的mask
        prob = p_pred_xy_oneimage[view_idx, :, 1]
        y_mat = prob.unsqueeze(1).expand(-1, p_img.shape[2])

        # cLine的mask
        def _genProb(probs):
            """
            把长为(angle_num,2)的，最后一维依次表示cup和cdown的霍夫域上的向量，拼接起来变为，cdown接在cup上的向量
            """
            # return torch.cat([probs[:, i] for i in range(probs.shape[1])]) # 该形式是原始的定义，与下面的完全等价但更复杂
            return probs.T.reshape(-1)

        prob = _genProb(p_pred_cud_oneimage[view_idx])
        vote_mask_c_up_down = get_vote_mask_c_up_down(cfg, p_img)
        c_mat = (prob * vote_mask_c_up_down).sum(-1) / vote_mask_c_up_down.sum(-1)
        result3 = torch.stack([x_mat, y_mat, c_mat])
        result_2dmask.append(result3)
    result_2dmask = torch.stack(result_2dmask)
    return result_2dmask


def _cvt_xyc_p_pred_2d_mask_to_wallceilfloor(cfg, result_2dmask, img_hw):
    # 将xyc红绿蓝转为竖直、天花板、地板红绿蓝
    zeros = result_2dmask.new_zeros(*img_hw)
    wallceilfloor_2dmask = []
    for view_idx, mask in enumerate(result_2dmask):
        if VIEW_NAME[view_idx] == "U":
            res = torch.stack([mask[2], (mask[0] + mask[1]) / 2, zeros], 0)
        elif VIEW_NAME[view_idx] == "D":
            res = torch.stack([mask[2], zeros, (mask[0] + mask[1]) / 2], 0)
        else:
            yAndC = (mask[1] + mask[2]) / 2
            halfHeight = yAndC.shape[0] // 2
            res = torch.stack([mask[0], torch.cat([yAndC[0:halfHeight], zeros[halfHeight:]], 0),
                               torch.cat([zeros[0:halfHeight], yAndC[halfHeight:]], 0)], 0)
        wallceilfloor_2dmask.append(res)
    wallceilfloor_2dmask = torch.stack(wallceilfloor_2dmask, 0)
    return wallceilfloor_2dmask


def _cal_p_pred_emask(cfg, result_2dmask, img_hw, e_img_hw):
    zeros = result_2dmask.new_zeros(result_2dmask.shape[1], *img_hw)
    # 将六个面的mask转到全景图上
    cube_mask = torch.cat(
        [torch.cat([zeros, result_2dmask[4], zeros, zeros], dim=2),
         torch.cat([*result_2dmask[[3, 0, 1, 2]]], dim=2),
         torch.cat([zeros, result_2dmask[5], zeros, zeros], dim=2)], dim=1)
    equal_mask = py360convert.c2e(cube_mask.permute(1, 2, 0).cpu().numpy(), *e_img_hw)
    equal_mask = torch.tensor(equal_mask, device="cpu").permute(2, 0, 1)  # c2e步骤后不再送回显卡，而是保持在CPU上供可视化等使用
    return equal_mask


def generatePred2DMask(cfg, input, output, img_idx):
    if "p_preds_2dmask" not in output: output["p_preds_2dmask"] = [None] * len(input["p_imgs"])
    if "p_preds_emask" not in output: output["p_preds_emask"] = [None] * len(input["p_imgs"])
    if output["p_preds_2dmask"][img_idx] is not None: return
    img_hw = input["p_imgs"][img_idx].shape[2:4]
    e_img_hw = input["e_img"][img_idx].shape[1:3]

    result_2dmask = _cal_p_pred_2d_mask(cfg, input, img_idx, torch.sigmoid(output["p_preds_xy"][img_idx]),
                                        torch.sigmoid(output["p_preds_cud"][img_idx]))

    wallceilfloor_2dmask = _cvt_xyc_p_pred_2d_mask_to_wallceilfloor(cfg, result_2dmask, img_hw)

    equal_mask = _cal_p_pred_emask(cfg, wallceilfloor_2dmask, img_hw, e_img_hw)

    output["p_preds_2dmask"][img_idx] = result_2dmask
    output["p_preds_emask"][img_idx] = equal_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg_file', type=str, required=True, help='specify the config for training')
    parser.add_argument('--ckpt', required=True, help='checkpoint for evaluation')
    parser.add_argument('--visu_count', default=20, type=int, help='visualize how many batches')
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')
    # Model related
    parser.add_argument('--backbone',
                        default='drn38',
                        choices=ENCODER_RESNET + ENCODER_DENSENET + ENCODER_HOUGH,
                        help='backbone of the network')
    parser.add_argument('--no_rnn', action='store_true', help='whether to remove rnn or not')
    parser.add_argument('--no_multigpus', action='store_true', help='disable data parallel')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    device = torch.device('cuda')

    dataset_valid = PerspectiveDataset(cfg, "test")
    loader_valid = DataLoader(dataset_valid,
                              args.batch_size,
                              collate_fn=dataset_valid.collate,
                              shuffle=False,
                              drop_last=False,
                              num_workers=0,
                              pin_memory=True)

    iterator_valid = iter(loader_valid)

    net = DMHNet(cfg, args.backbone, not args.no_rnn).to(device)
    if not args.no_multigpus:
        net = nn.DataParallel(net)  # multi-GPU

    print(str(cfg.POST_PROCESS))

    state_dict = pipeload(args.ckpt, map_location='cpu')["state_dict"]
    net.load_state_dict(state_dict, strict=True)
    net.eval()

    count = 0
    DBG_START = 0

    for valid_idx in trange(args.visu_count, desc='PostProcess Visualization', position=2):
        input = next(iterator_valid)
        with torch.no_grad():
            for k in input:
                if isinstance(input[k], torch.Tensor):
                    input[k] = input[k].to(device)
            _, results_dict = net(input)

            for i in range(len(input["filename"])):
                count += 1
                if count <= DBG_START:
                    continue
                (gt_lines, gt_lwh, gt_cor_id), (pred_lines, pred_lwh, pred_cor_id), metric = postProcess(cfg, input,
                                                                                                         results_dict,
                                                                                                         i)
                print("{:s} pred{:s} gt{:s}".format(str(metric), str(pred_lwh), str(gt_lwh)))

                # 画方框的代码
                from visualization import o3dRunVis, o3dDrawLines, o3dInitVis

                vis = o3dInitVis()
                o3dDrawLines(vis, gt_lines, gt_lwh, [1.0, 0.0, 0.0])
                o3dDrawLines(vis, pred_lines, pred_lwh)
                o3dRunVis(vis)

                # 画角点的代码
                from visualization import drawEqualRectCorners

                drawEqualRectCorners(plt, input["e_img"][i], gt_cor_id, pred_cor_id)
                plt.show()
