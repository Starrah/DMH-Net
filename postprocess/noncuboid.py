import traceback
import warnings
from typing import List, Tuple

import numpy as np
import torch
from easydict import EasyDict
from torch import nn

from e2pconvert_torch.e2plabelconvert import generateOnePerspectiveLabel
from e2pconvert_torch.torch360convert import coor2uv, xyz2uv, uv2unitxyz, uv2coor
from e2plabel.e2plabelconvert import VIEW_NAME, VIEW_ARGS, VIEW_SIZE
from eval_general import test_general
from perspective_dataset import PerspectiveDataset
from postprocess.GDSolver import solve
from postprocess.LayoutNetv2 import map_coordinates_Pytorch
from postprocess.postprocess2 import generatePred2DMask, _cal_p_pred_emask, findPeaks, \
    lossFunction, calMetrics_PretendDisUThenOptimIOUForV2

FILE_LIST = [
    "7y3sRwLe3Va_1679b5de39e548d38ba240f2fd99cae9.png",
    "7y3sRwLe3Va_1679b5de39e548d38ba240f2fd99cae9.png",
    "7y3sRwLe3Va_b564162b2c7d4033bfe6ef3dfb959c9e.png",
    "7y3sRwLe3Va_fdab6422162e49db822a37178ab70481.png",
    "B6ByNegPMKs_0e5ba44387774783903fea2a1b8f53dd.png",
    "B6ByNegPMKs_4c769d1a658d41eb995deb5b40af57a4.png",
    "7y3sRwLe3Va_5c39473b25b74307858764d1a2045b9e.png",
    "7y3sRwLe3Va_6376b741b50a4418b3dc3fde791c3c09.png",
]


def nonCuboidPostProcess(cfg, input, output, img_idx):
    if cfg.get("VISUALIZATION", {}).get("TYPE") is None:
        cfg.VISUALIZATION = EasyDict()
        cfg.VISUALIZATION.TYPE = [["c", "y", "x", "e_rm", "gtlines", "text"]]
    generatePred2DMask(cfg, input, output, img_idx)
    mask2ds = output["p_preds_2dmask"][img_idx]
    gt_cor_id = input["cor"][img_idx]
    gt_cor_id_np = gt_cor_id.cpu().numpy()

    cor_mask2ds = torch.stack([
        mask2ds[:, 0] * mask2ds[:, 1],
        mask2ds[:, 1] * mask2ds[:, 2],
        mask2ds[:, 0] * mask2ds[:, 2],
    ], 1)
    if cfg.POST_PROCESS.get("COR_IMG_CAL") is None:
        cor_mask2ds = cor_mask2ds.mean(1)
    elif cfg.POST_PROCESS.get("COR_IMG_CAL") == "max":
        cor_mask2ds = cor_mask2ds.max(1)[0]
    elif cfg.POST_PROCESS.get("COR_IMG_CAL") == "merge":
        cor_mask2ds = (cor_mask2ds.sum(1) + cor_mask2ds.max(1)[0] * 2) / 5
    img_hw = input["p_imgs"][img_idx].shape[2:4]
    e_img_hw = input["e_img"][img_idx].shape[1:3]
    cor_img = _cal_p_pred_emask(cfg, cor_mask2ds.unsqueeze(1), img_hw, e_img_hw).squeeze(0)
    column_img = cor_img.max(0)[0]

    columns = []  # 长度为n的数组，前4、6、8...代表4、6、8时的初始化
    ranges = [(e_img_hw[1] // 4 * i, e_img_hw[1] // 4 * (i + 1)) for i in range(4)]
    scores = [column_img[r[0]:r[1]] for r in ranges]
    peakss = [findPeaks(cfg, r).to(torch.int64) for r in scores]
    peakss = [r[s[r].argsort(descending=True)] for r, s in zip(peakss, scores)]
    # 先从每个四分之一图找一个点
    columns.extend([(r[0] if len(r) > 0 else r.new_tensor(e_img_hw[1] // 4 // 4)) + begin for r, (begin, _) in
                    zip(peakss, ranges)])
    peakss_2 = torch.cat([r[1:] + begin for r, (begin, _) in zip(peakss, ranges)])
    # 剩下的中按置信度排序
    peakss_2 = peakss_2[column_img[peakss_2].argsort(descending=True)]
    columns.extend(peakss_2)
    columns = torch.stack(columns)

    res_v2 = None

    all_results = []
    # for cor_num in [6]:
    METHODS = []
    METHODS.extend([(n, "v2") for n in cfg.POST_PROCESS.get("COMBINE", {}).get("V2", [4,6,8,10,12])])
    METHODS.extend([(n, "v1") for n in cfg.POST_PROCESS.get("COMBINE", {}).get("V1", [4,6,8,10,12])])
    for cor_num, method in METHODS:
        # TODO: 设置length恒为1，看看到底是直线好还是线段好；
        # TODO: 设置loss聚合的方法，比如求和？平均？
        if method == "v0":
            if len(columns) < cor_num: continue
            columns_one = columns[0:cor_num].sort()[0]  # 记得一定要排序！！！！

            cor_img_columns = cor_img[:, columns_one]
            upper_y = cor_img_columns[:img_hw[0] // 2].argmax(0)
            lower_y = cor_img_columns[img_hw[0] // 2:].argmax(0) + (img_hw[0] // 2)
            init_cors = torch.cat(
                [cor_img.new_tensor([[c, u], [c, l]]) for c, u, l in zip(columns_one, upper_y, lower_y)], 0)

            # cornersToPeaks(cfg, input, img_idx, init_cors)
            # pred_cor_id=init_cors
            # metrics = {}

            pred_cor_id, err_score = solveCorners(cfg, input, output, img_idx, init_cors, e_img_hw)
        else:
            if method == "v2":
                if res_v2 is None:
                    res_v2 = generate2DFrameFromColumnPeaksV2(cfg, columns, cor_img, ranges)
                res_v2_0, init_cors_all = res_v2  # init_cors现在由V2主函数返回
                res_frame = generate2DFrameFromColumnPeaksV2_ChooseCorNum(res_v2_0, columns, cor_num, ranges)
                if res_frame is None:
                    continue
                choice_idx, beginFromZ = res_frame
                choice_corner_idx = torch.cat([torch.stack([2 * v, 2 * v + 1]) for v in choice_idx])
                init_cors = init_cors_all[choice_corner_idx]
            elif method == "v1":
                res_frame = generate2DFrameFromColumnPeaks(columns, cor_img, cor_num)
                if res_frame is None:
                    continue
                choice_idx, beginFromZ = res_frame
                columns_one = columns[choice_idx]
                cor_img_columns = cor_img[:, columns_one]
                upper_y = cor_img_columns[:img_hw[0] // 2].argmax(0)
                lower_y = cor_img_columns[img_hw[0] // 2:].argmax(0) + (img_hw[0] // 2)
                init_cors = torch.cat(
                    [cor_img.new_tensor([[c, u], [c, l]]) for c, u, l in zip(columns_one, upper_y, lower_y)], 0)

            # pred_cor_id = init_cors
            if not cfg.POST_PROCESS.get("COMBINE", {}).get("SOLVE3", False):
                pred_cor_id, err_score = solveCorners2(cfg, input, output, img_idx, init_cors, cor_img, beginFromZ,
                                                       e_img_hw)
            else:
                pred_cor_id, err_score = solveCorners3(cfg, input, output, img_idx, init_cors, cor_img, beginFromZ,
                                                       e_img_hw)

            # cornersToPeaks(cfg, input, img_idx, pred_cor_id)
            # metrics = {}

        pred_cor_id_np = pred_cor_id.cpu().numpy()

        one_result = {"cor_num": cor_num, "err_score": err_score, "pred_cor": pred_cor_id, "method": method}
        metric = {}
        # 调用HorizonNet的算指标代码进行计算
        t = {}
        if cfg.DATA.TYPE == "cuboid":
            assert False
            test(pred_cor_id_np, z0, z1, gt_cor_id_np, e_img_hw[1], e_img_hw[0], t)
        elif cfg.DATA.TYPE == "general":
            test_general(pred_cor_id_np, gt_cor_id_np, e_img_hw[1], e_img_hw[0], t)
        else:
            assert False

        for k in t:
            metric[k] = torch.tensor(t[k]) if not isinstance(t[k], str) else t[k]

        one_result["metrics"] = metric
        all_results.append(one_result)

    if cfg.POST_PROCESS.get("COMBINE", {}).get("OPTIM", True):
        _, (_, _, pred_cor_id_optim), metric_optim, err_score_optim = \
            calMetrics_PretendDisUThenOptimIOUForV2(cfg, input, output, img_idx, True)
        all_results.append({"cor_num": 4, "err_score": err_score_optim, "pred_cor": pred_cor_id_optim, "method": "optim", "metrics": metric_optim})

    METRIC_KEYS=["2DIoU", "3DIoU", "rmse", "delta_1"]
    err_scores = torch.tensor([one_result["err_score"] for one_result in all_results])
    best_result_idx = err_scores.argmin()
    best_result_cor_num = all_results[best_result_idx]["cor_num"]
    pred_cor_id = all_results[best_result_idx]["pred_cor"]
    metrics = {}
    metrics.update(all_results[best_result_idx]["metrics"])

    # all_results中直接挑指标最好的
    metrics_all = {k: torch.stack([one_result["metrics"][k] for one_result in all_results]) for k in METRIC_KEYS}
    metrics_best = {"best/" + k: v.max() for k, v in metrics_all.items()}
    metrics.update(metrics_best)

    metrics["pred_cor_num"] = torch.tensor(float(best_result_cor_num))

    # 但是，仍然要打印所有的结果
    additional_metrics = {}
    for one_result in all_results:
        additional_metrics.update(
            {one_result["method"] + "/" + str(one_result["cor_num"]) + "/" + k: one_result["metrics"][k] for k in one_result["metrics"]})
        additional_metrics.update({one_result["method"] + "/" + str(one_result["cor_num"]) + "/err_score": one_result["err_score"]})
    print(additional_metrics)

    return (None, None, gt_cor_id), (None, None, pred_cor_id), metrics


def cornersToPeaks(cfg, input, img_idx, cors, isInputUv=False) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
    result = []
    for view_idx, (view_name, view) in enumerate(zip(VIEW_NAME, VIEW_ARGS)):
        r = generateOnePerspectiveLabel(input["e_img"][img_idx], cors, *view, VIEW_SIZE, isInputUv=isInputUv)
        # input["lines"][img_idx][view_idx] = r["lines"]  # TODO
        peakss, lengths = PerspectiveDataset.linesToPeaksNewCore([line[3:8] for line in r["lines"]], VIEW_SIZE)
        xPeaks, yPeaks, cUpPeaks, cDownPeaks = ((torch.stack(p) if len(p) > 0 else cors.new_zeros(0)) for p in peakss)
        xLength, yLength, cUpLength, cDownLength = ((torch.stack(p) if len(p) > 0 else cors.new_zeros(0)) for p in
                                                    lengths)
        result.append([(xPeaks, xLength), (yPeaks, yLength), (cUpPeaks, cUpLength), (cDownPeaks, cDownLength)])
    return result


def solveCorners(cfg, input, output, img_idx, start_corners, e_img_hw):
    """
    参数定义：共(n*2+1)个，其中n是4,6,8...(len(start_corners)=2*n)
    分别表示角点的x值、天花板的tan值、地板tan除以天花板tan。
    """
    uv = coor2uv(start_corners, *e_img_hw)
    xs = start_corners[::2, 0]
    tanceil = torch.tan(uv[::2, 1])
    tanratio = (torch.tan(uv[1::2, 1]) / torch.tan(uv[::2, 1])).mean()

    class CornersSolveModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.xs = nn.Parameter(xs.clone().detach().requires_grad_(True))
            self.tanceil = nn.Parameter(tanceil.clone().detach().requires_grad_(True))
            self.tanratio = nn.Parameter(tanratio.clone().detach().requires_grad_(True))

        def toCors(self):
            tanfloor = self.tanceil * self.tanratio
            tanceilfloor = []
            for c, f in zip(self.tanceil, tanfloor):
                tanceilfloor.extend((c, f))
            vs = torch.atan(torch.stack(tanceilfloor))
            coor_y = (-vs / np.pi + 0.5) * e_img_hw[0] - 0.5
            cors = torch.stack([self.xs.repeat_interleave(2), coor_y], 1)
            return cors

        def forward(self):
            cors = self.toCors()
            try:
                peaks = cornersToPeaks(cfg, input, img_idx, cors)
                return lossFunction(output, img_idx, peaks)
            except:
                traceback.print_exc()
                warnings.warn("CornersSolveModule forward时抛出异常")
                return torch.tensor(20., device=cors.device, requires_grad=True)

    module = CornersSolveModule().to("cpu")
    module = solve(module, max_iter=100, lr=1e-2, stop_tol=1e-3, stop_range=5)
    return module.toCors(), torch.abs(module.forward())


def generate2DFrameFromColumnPeaksV2(cfg, column_peaks, cor_img, ranges):
    """
    :param column_peaks (x) 不定长vector。所有的列peak，要求按照置信度从大到小排好序；且其中前四个元素必须恰好位于四个方向上。
    :return None或Tuple[Tensor<cor_num>, bool]。Tensor<cor_num>是column_peaks中应被选中的元素的 **下标**，bool表示chosen[0]到chosen[1]之间的连线应该沿z（前后）方向。
    """

    # 只用前4个点，求出一个假想高度，把所有peak投射到xz空间里去
    column_peaks, sortedIdxs = column_peaks.sort()
    oldIdxToNewIdx = torch.cat([torch.where(sortedIdxs == v)[0] for v in range(len(sortedIdxs))])

    def getXz():
        img_hw = cor_img.shape
        columns_one = column_peaks
        cor_img_columns = cor_img[:, columns_one]
        upper_y = cor_img_columns[:img_hw[0] // 2].argmax(0)
        lower_y = cor_img_columns[img_hw[0] // 2:].argmax(0) + (img_hw[0] // 2)
        init_cors = torch.cat(
            [cor_img.new_tensor([[c, u], [c, l]]) for c, u, l in zip(columns_one, upper_y, lower_y)], 0)

        uv = coor2uv(init_cors, *img_hw)
        tanv = torch.tan(uv[:, 1])
        # xs = start_corners[::2, 0]
        # tanceil = torch.tan(uv[::2, 1])
        # 第一步：计算height，并把所有的点都只移动v坐标调整到同一高度上。
        tanratio = tanv[0::2] / tanv[1::2]
        # ！！！只用传进来的前4个点做支撑点！注意要用oldIdxToNewIdx做idx转换
        if cfg.POST_PROCESS.V2.get("REMOVE_BAD_GROUND_POINT") == "both":
            prob_ground = map_coordinates_Pytorch(cor_img, init_cors[1::2][oldIdxToNewIdx[0:4]].T.flip(0))
            the_mask = chooseByProb(cfg, prob_ground)
            aim_tanratio = tanratio[oldIdxToNewIdx[0:4][the_mask]].mean()
        else:
            aim_tanratio = tanratio[oldIdxToNewIdx[0:4]].mean()
        v_adjust_factor = torch.sqrt(tanratio / aim_tanratio)
        v_adjust_vector = torch.cat([torch.stack([1 / v, v]) for v in v_adjust_factor])
        tanv = tanv * v_adjust_vector
        uv[:, 1] = torch.atan(tanv)
        # 第二步：构建初始边框
        # 所有的点uv已知，投影到xyz上、固定y轴的值为height或-1.6，得到x和z
        # 根据beginFromZ和置信度，refine每个点的x坐标和z坐标。
        xyz = uv2unitxyz(uv)
        # xyz[0::2] = xyz[0::2] / xyz[0::2, 1] * height
        # xyz[1::2] = xyz[1::2] / xyz[1::2, 1] * -1.6
        xz = (xyz[1::2] / xyz[1::2, 1:2] * -1.6)[:, [0, 2]]

        e_coor = uv2coor(uv, *cor_img.shape)
        prob = map_coordinates_Pytorch(cor_img, e_coor.T.flip(0))
        prob = (prob[0::2] + prob[1::2]) / 2

        # TODO 问题：用进入下一阶段的点坐标用什么？refine过一轮的？还是没refine的？
        coor = e_coor if cfg.POST_PROCESS.V2.REFINE_V_TWICE else init_cors
        oldIdxToNewCornerIdx = torch.cat([torch.stack([2 * v, 2 * v + 1]) for v in oldIdxToNewIdx])
        coor = coor[oldIdxToNewCornerIdx]
        return xz, prob, coor

    xz, prob, coor = getXz()

    # TODO 问题：用什么概率？是新的点概率，还是列概率？
    if cfg.POST_PROCESS.V2.FIRST_COLUMN_PROB:
        column_img = cor_img.max(0)[0]
        prob = column_img[column_peaks]

    l = len(xz)
    # 逐点连线计算方向
    delta = xz - torch.cat([xz[1:], xz[0:1]])
    absdelta = torch.abs(delta)
    lineIsZ: torch.Tensor = (absdelta[:, 0] < absdelta[:, 1])  # 从i到i+1的线是否沿z方向
    # 选择【概率最高的、两边连线是x和z方向】的点开始
    probIdx = prob.argsort(descending=True)

    resultIdxs = None
    for v in probIdx:
        if lineIsZ[(v - 1 + l) % l] == lineIsZ[v]: continue  # 两侧直线方向相同，无效点
        beginIdx = v.item()

        resultIdxs = [beginIdx]
        # 从beginIndex起，沿哪个方向遍历结果都是一样的，所以固定沿正方向走
        cur = (beginIdx + 1) % l
        beginFromZ = lineIsZ[beginIdx]
        curDirect = beginFromZ
        while cur != beginIdx:
            if lineIsZ[cur] != curDirect:
                resultIdxs.append(cur)
                curDirect = lineIsZ[cur]
            cur = (cur + 1) % l

        # 验证操作的结果：只有操作后仍能保证全景图四个区域都还有peak，才准这么做
        new_choice_peaks = column_peaks[sortedIdxs[resultIdxs]]
        success = True
        for r in ranges:
            if torch.logical_and(r[0] <= new_choice_peaks, new_choice_peaks < r[1]).sum() <= 0:
                success = False
                break
        if success:
            break
        else:
            resultIdxs = None

    if resultIdxs is None:
        # 如果找不到任何可行解，则只能拿前四个点跑一遍算了
        if len(column_peaks) > 4:
            res_v2_0, _ = generate2DFrameFromColumnPeaksV2(cfg, column_peaks[oldIdxToNewIdx][0:4], cor_img, ranges)
            return res_v2_0, coor
        else:
            # 递归后仍然不行，那就直接返回吧
            resultIdxs = [0, 1, 2, 3]
            beginFromZ = lineIsZ[0]

    return (sortedIdxs[resultIdxs], prob[resultIdxs], beginFromZ), coor


def generate2DFrameFromColumnPeaksV2_ChooseCorNum(res_v2_0, column_peaks, cor_num, ranges):
    choice_idx, choice_probs, beginFromZ = res_v2_0
    assert len(choice_idx) % 2 == 0 and 4 <= cor_num <= 12 and cor_num % 2 == 0
    if len(choice_idx) == cor_num:
        return choice_idx, beginFromZ
    elif len(choice_idx) < cor_num:
        return None
    else:
        # 一次只删两个点，然后递归
        # 寻找概率最低的线段，remove
        line_prob = choice_probs + torch.cat([choice_probs[1:], choice_probs[0:1]])
        idxSeq = line_prob.argsort()
        new_choice_idx, new_choice_probs, new_beginFromZ = None, None, None
        # 选择要delete的线：
        for toRemoveIdx in idxSeq:
            if toRemoveIdx < len(choice_idx) - 1:
                # 直接两端拼接即可
                new_choice_idx = torch.cat([choice_idx[0:toRemoveIdx], choice_idx[toRemoveIdx + 2:]])
                new_choice_probs = torch.cat([choice_probs[0:toRemoveIdx], choice_probs[toRemoveIdx + 2:]])
                new_beginFromZ = beginFromZ
            else:
                # 从1到l-1，且beginFromZ要反转
                new_choice_idx = choice_idx[1:len(choice_idx) - 1]
                new_choice_probs = choice_probs[1:len(choice_idx) - 1]
                new_beginFromZ = not beginFromZ
            # 验证操作的结果：只有操作后仍能保证全景图四个区域都还有peak，才准这么做
            new_choice_peaks = column_peaks[new_choice_idx]
            success = True
            for r in ranges:
                if torch.logical_and(r[0] <= new_choice_peaks, new_choice_peaks < r[1]).sum() <= 0:
                    success = False
                    break
            if success:
                # 如果验证通过，则说明成功减掉了两个点。
                # 则递归下去继续减点，如果下层返回了值就是通过了，如果下层返回了None，则本层继续
                new_res_v2_0 = (new_choice_idx, new_choice_probs, new_beginFromZ)
                res = generate2DFrameFromColumnPeaksV2_ChooseCorNum(new_res_v2_0, column_peaks, cor_num, ranges)
                if res is not None:
                    return res
        # 如果走到这里还没成功递归出有效的结果，就是不行了
        return None


def generate2DFrameFromColumnPeaks(column_peaks, cor_img, cor_num):
    """
    :param column_peaks (x) 不定长vector。所有的列peak，要求按照置信度从大到小排好序；且其中前四个元素必须恰好位于四个方向上。
    :return None或Tuple[Tensor<cor_num>, bool]。Tensor<cor_num>是column_peaks中应被选中的元素的 **下标**，bool表示chosen[0]到chosen[1]之间的连线应该沿z（前后）方向。
    """
    assert 4 <= cor_num <= 12 and cor_num % 2 == 0
    toChooseIdxs = list(range(4, len(column_peaks)))
    chosen = list(column_peaks[0:4].sort()[0])
    beginFromZ = True  # chosen[0]到chosen[1]之间的连线应该沿z（前后）方向
    while len(chosen) < cor_num:
        if len(toChooseIdxs) < 2:
            return None
        beforeChosenCount = len(chosen)
        idx0 = toChooseIdxs[0]
        for i in range(1, len(toChooseIdxs)):
            idx1 = toChooseIdxs[i]
            insert_pos = torch.searchsorted(column_peaks.new_tensor(chosen), column_peaks[[idx0, idx1]])
            insert_pos_dis = torch.abs(insert_pos[1] - insert_pos[0])
            if insert_pos_dis <= 1 or insert_pos_dis == len(chosen):
                if insert_pos_dis == len(chosen):
                    # 环形插入的，则beginFromY应反转
                    beginFromZ = not beginFromZ
                if column_peaks[idx1] >= column_peaks[idx0]:
                    # idx1位置处元素的插入位置比idx0位置处元素靠后，先插入idx1位置处元素
                    chosen.insert(insert_pos[1], column_peaks[idx1])
                    chosen.insert(insert_pos[0], column_peaks[idx0])
                else:
                    chosen.insert(insert_pos[0], column_peaks[idx0])
                    chosen.insert(insert_pos[1], column_peaks[idx1])
                toChooseIdxs.remove(idx0)
                toChooseIdxs.remove(idx1)
                break
        if len(chosen) - beforeChosenCount < 2:
            # 尝试插入失败，则应该返回None
            return None
    return torch.cat([torch.where(column_peaks == v)[0] for v in chosen]), beginFromZ


def chooseByProb(cfg, prob):
    MIN_COUNT = 2
    if cfg.POST_PROCESS.get("COR_IMG_CAL") is None:
        VALUE = 0.4
    elif cfg.POST_PROCESS.get("COR_IMG_CAL") == "max":
        VALUE = 0.7
    elif cfg.POST_PROCESS.get("COR_IMG_CAL") == "merge":
        VALUE = 0.6
    result = prob > VALUE
    result[prob.argsort(descending=True)[0:MIN_COUNT]] = True
    return result


def solveCorners2(cfg, input, output, img_idx, start_corners: torch.Tensor, prob_map: torch.Tensor,
                  beginFromZ: bool, e_img_hw):
    uv = coor2uv(start_corners, *e_img_hw)
    tanv = torch.tan(uv[:, 1])
    # xs = start_corners[::2, 0]
    # tanceil = torch.tan(uv[::2, 1])
    # 第一步：计算height，并把所有的点都只移动v坐标调整到同一高度上。
    tanratio = tanv[0::2] / tanv[1::2]
    if cfg.POST_PROCESS.V2.get("REMOVE_BAD_GROUND_POINT") == "second" or cfg.POST_PROCESS.V2.get("REMOVE_BAD_GROUND_POINT") == "both":
        prob_ground = map_coordinates_Pytorch(prob_map, start_corners[1::2].T.flip(0))
        the_mask = chooseByProb(cfg, prob_ground)
        aim_tanratio = tanratio[the_mask].mean()
    else:
        aim_tanratio = tanratio.mean()
    height = -1.6 * aim_tanratio  # 根据(天花板tan/地板tan)的平均值确定初始高度
    v_adjust_factor = torch.sqrt(tanratio / aim_tanratio)
    v_adjust_vector = torch.cat([torch.stack([1 / v, v]) for v in v_adjust_factor])
    tanv = tanv * v_adjust_vector
    uv[:, 1] = torch.atan(tanv)
    # 第二步：构建初始边框
    # 所有的点uv已知，投影到xyz上、固定y轴的值为height或-1.6，得到x和z
    # 根据beginFromZ和置信度，refine每个点的x坐标和z坐标。
    xyz = uv2unitxyz(uv)
    # xyz[0::2] = xyz[0::2] / xyz[0::2, 1] * height
    # xyz[1::2] = xyz[1::2] / xyz[1::2, 1] * -1.6
    xz = (xyz[1::2] / xyz[1::2, 1:2] * -1.6)[:, [0, 2]]

    e_coor = uv2coor(uv, *prob_map.shape)
    if cfg.POST_PROCESS.V2.SECOND_START_PROB:
        prob_full = map_coordinates_Pytorch(prob_map, start_corners.T.flip(0))
    else:
        prob_full = map_coordinates_Pytorch(prob_map, e_coor.T.flip(0))
    prob = (prob_full[0::2] + prob_full[1::2]) / 2

    dis = torch.zeros_like(xz[:, 0])
    l = len(dis)
    for i in range(l):
        if cfg.POST_PROCESS.V2.STRAIGHTEN_BY_PROB:
            factor = torch.stack([prob[i], prob[(i + 1) % l]])
            factor = factor / factor.sum()
            if cfg.POST_PROCESS.V2.STRAIGHTEN_WHEN_BETTER:
                if factor[0] > 0.75:
                    factor = factor.new_tensor([1.0, 0.0])
                elif factor[1] > 0.75:
                    factor = factor.new_tensor([0.0, 1.0])
        else:
            factor = prob.new_tensor([0.5, 0.5])
        if beginFromZ:
            # xz[0]到xz[1]是沿z方向的，所以应该强迫i=0时x值相等，填入dis[0]处
            if i % 2 == 0:
                dis[i] = (torch.stack([xz[i, 0], xz[(i + 1) % l, 0]]) * factor).sum()
            else:
                dis[i] = (torch.stack([xz[i, 1], xz[(i + 1) % l, 1]]) * factor).sum()
        else:
            # 否则，xz[0]到xz[1]是沿x方向的，则应该强迫i=0时z值相等，填入1处
            if i % 2 == 0:
                dis[(i + 1) % l] = (torch.stack([xz[i, 1], xz[(i + 1) % l, 1]]) * factor).sum()
            else:
                dis[(i + 1) % l] = (torch.stack([xz[i, 0], xz[(i + 1) % l, 0]]) * factor).sum()

    class CornersSolveModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.height = nn.Parameter(height.clone().detach().requires_grad_(True))
            self.dis = nn.Parameter(dis.clone().detach().requires_grad_(True))

        def toUvs(self):
            cor_num = len(self.dis)
            idxs = torch.arange(cor_num, device=self.dis.device)
            idxs = torch.stack([idxs, (idxs + 1) % cor_num], 1)
            idxs[1::2] = idxs[1::2].flip(1)
            xyz = torch.cat([
                torch.stack([self.dis[idxs[:, 0]], self.height.repeat(len(idxs)),
                             self.dis[idxs[:, 1]]], 1),
                torch.stack([self.dis[idxs[:, 0]], self.dis.new_tensor(-1.6).repeat(len(idxs)),
                             self.dis[idxs[:, 1]]], 1),
            ], 0)
            uv = xyz2uv(xyz)
            # 进行排序，这步是必须的！
            seq = uv[:cor_num, 0].argsort()
            seq = seq.repeat_interleave(2)
            seq[1::2] += cor_num
            uv = uv[seq]
            return uv

        def forward(self):
            # return torch.tensor(20., device=self.dis.device, requires_grad=True)
            uv = self.toUvs()
            try:
                peaks = cornersToPeaks(cfg, input, img_idx, uv, isInputUv=True)
                return lossFunction(output, img_idx, peaks)
            except:
                traceback.print_exc()
                warnings.warn("CornersSolveModule forward时抛出异常")
                return torch.tensor(20., device=uv.device, requires_grad=True)

    module = CornersSolveModule().to("cpu")
    module = solve(module, max_iter=100, lr=1e-2, stop_tol=1e-3, stop_range=5)
    uvs = module.toUvs()
    return uv2coor(uvs, *e_img_hw), torch.abs(module.forward())


def solveCorners3(cfg, input, output, img_idx, start_corners: torch.Tensor, prob_map: torch.Tensor,
                  beginFromZ: bool, e_img_hw):
    uv = coor2uv(start_corners, *e_img_hw)
    tanv = torch.tan(uv[:, 1])
    # xs = start_corners[::2, 0]
    # tanceil = torch.tan(uv[::2, 1])
    # 第一步：计算height，并把所有的点都只移动v坐标调整到同一高度上。
    tanratio = tanv[0::2] / tanv[1::2]
    if cfg.POST_PROCESS.V2.get("REMOVE_BAD_GROUND_POINT") == "second" or cfg.POST_PROCESS.V2.get("REMOVE_BAD_GROUND_POINT") == "both":
        prob_ground = map_coordinates_Pytorch(prob_map, start_corners[1::2].T.flip(0))
        the_mask = chooseByProb(cfg, prob_ground)
        aim_tanratio = tanratio[the_mask].mean()
    else:
        aim_tanratio = tanratio.mean()
    height = -1.6 * aim_tanratio  # 根据(天花板tan/地板tan)的平均值确定初始高度
    v_adjust_factor = torch.sqrt(tanratio / aim_tanratio)
    v_adjust_vector = torch.cat([torch.stack([1 / v, v]) for v in v_adjust_factor])
    tanv = tanv * v_adjust_vector
    uv[:, 1] = torch.atan(tanv)
    # 第二步：构建初始边框
    # 所有的点uv已知，投影到xyz上、固定y轴的值为height或-1.6，得到x和z
    # 根据beginFromZ和置信度，refine每个点的x坐标和z坐标。
    xyz = uv2unitxyz(uv)
    # xyz[0::2] = xyz[0::2] / xyz[0::2, 1] * height
    # xyz[1::2] = xyz[1::2] / xyz[1::2, 1] * -1.6
    xz = (xyz[1::2] / xyz[1::2, 1:2] * -1.6)[:, [0, 2]]

    e_coor = uv2coor(uv, *prob_map.shape)
    if cfg.POST_PROCESS.V2.SECOND_START_PROB:
        prob_full = map_coordinates_Pytorch(prob_map, start_corners.T.flip(0))
    else:
        prob_full = map_coordinates_Pytorch(prob_map, e_coor.T.flip(0))
    prob = (prob_full[0::2] + prob_full[1::2]) / 2

    dis = torch.zeros_like(xz[:, 0])
    l = len(dis)
    for i in range(l):
        if cfg.POST_PROCESS.V2.STRAIGHTEN_BY_PROB:
            factor = torch.stack([prob[i], prob[(i + 1) % l]])
            factor = factor / factor.sum()
            if cfg.POST_PROCESS.V2.STRAIGHTEN_WHEN_BETTER:
                if factor[0] > 0.75:
                    factor = factor.new_tensor([1.0, 0.0])
                elif factor[1] > 0.75:
                    factor = factor.new_tensor([0.0, 1.0])
        else:
            factor = prob.new_tensor([0.5, 0.5])
        if beginFromZ:
            # xz[0]到xz[1]是沿z方向的，所以应该强迫i=0时x值相等，填入dis[0]处
            if i % 2 == 0:
                dis[i] = (torch.stack([xz[i, 0], xz[(i + 1) % l, 0]]) * factor).sum()
            else:
                dis[i] = (torch.stack([xz[i, 1], xz[(i + 1) % l, 1]]) * factor).sum()
        else:
            # 否则，xz[0]到xz[1]是沿x方向的，则应该强迫i=0时z值相等，填入1处
            if i % 2 == 0:
                dis[(i + 1) % l] = (torch.stack([xz[i, 1], xz[(i + 1) % l, 1]]) * factor).sum()
            else:
                dis[(i + 1) % l] = (torch.stack([xz[i, 0], xz[(i + 1) % l, 0]]) * factor).sum()

    class CornersSolveModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.height = nn.Parameter(height.clone().detach().requires_grad_(True))
            self.dis = nn.Parameter(dis.clone().detach().requires_grad_(True))

        def toUvs(self):
            cor_num = len(self.dis)
            idxs = torch.arange(cor_num, device=self.dis.device)
            idxs = torch.stack([idxs, (idxs + 1) % cor_num], 1)
            idxs[1::2] = idxs[1::2].flip(1)

            dis = self.dis
            disU = torch.atan2(dis[0::2], dis[1::2]) # atan2(x/z)
            seq = disU.argsort()
            seq = (seq * 2).repeat_interleave(2)
            seq[1::2] += 1
            dis = dis[seq]

            xyz = torch.cat([
                torch.stack([dis[idxs[:, 0]], self.height.repeat(len(idxs)),
                             dis[idxs[:, 1]]], 1),
                torch.stack([dis[idxs[:, 0]], dis.new_tensor(-1.6).repeat(len(idxs)),
                             dis[idxs[:, 1]]], 1),
            ], 0)
            uv = xyz2uv(xyz)

            # 不再排序，而是直接按照uv中的既定顺序，只是找到u最小的点从这里开始
            startPlace = uv[0:cor_num].argmin()
            seq = torch.arange(startPlace, startPlace + cor_num, device=uv.device) % cor_num
            seq = seq.repeat_interleave(2)
            seq[1::2] += cor_num
            uv = uv[seq]
            return uv

        def forward(self):
            # return torch.tensor(20., device=self.dis.device, requires_grad=True)
            uv = self.toUvs()
            try:
                peaks = cornersToPeaks(cfg, input, img_idx, uv, isInputUv=True)
                return lossFunction(output, img_idx, peaks)
            except:
                traceback.print_exc()
                warnings.warn("CornersSolveModule forward时抛出异常")
                return torch.tensor(20., device=uv.device, requires_grad=True)

    module = CornersSolveModule().to("cpu")
    module = solve(module, max_iter=100, lr=1e-2, stop_tol=1e-3, stop_range=5)
    uvs = module.toUvs()
    return uv2coor(uvs, *e_img_hw), torch.abs(module.forward())
