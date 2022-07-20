import io
import math
import os
import time
from typing import Dict

import cv2
import numpy as np

try:
    import open3d as o3d
except:
    pass
import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from e2plabel.e2plabelconvert import VIEW_NAME
from perspective_dataset import PerspectiveDataset
from postprocess.postprocess2 import postProcess, get_vote_mask_c_up_down, generatePred2DMask, _cal_p_pred_emask

DRAW_CUBE_POSITIONS = {
    "F": [1, 1],
    "R": [1, 2],
    "B": [1, 3],
    "L": [1, 0],
    "U": [0, 1],
    "D": [2, 1],
    "E": [0, 1, 2, 4],
    "3D": [2, 3, 2, 4],
    "TEXT": [0, 0]
}

DEFAULT_DRAWTYPE = [["c", "y", "x", "gtlines", "e_rm", "3d", "text"]]


# DEFAULT_DRAWTYPE = [['gtlines_colored', 'border'], 'e_gt', ['c_cl2', 'y_cl2', 'x_cl2']] # GT可视化（论文图1）所用的配置


def clearAxesLines(ax: plt.Axes):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def getCubeAxes(fig: plt.Figure, view_name):
    spec = fig.add_gridspec(3, 4, hspace=0, wspace=0)
    posi = DRAW_CUBE_POSITIONS[view_name]
    if len(posi) == 2:
        ax: plt.Axes = fig.add_subplot(spec[posi[0], posi[1]])
    else:
        ax: plt.Axes = fig.add_subplot(spec[posi[0]:posi[1], posi[2]:posi[3]])
    clearAxesLines(ax)
    return ax


def getMaskByType(type, cfg, input, output, img_idx, view_idx):
    """
    对每个面画线
    """
    p_img = input["p_imgs"][img_idx, view_idx]
    prob = None

    if type.find("x") == 0:
        color_type = 0
        if type.find("cl2") != -1: # 论文图1的Line Predictions中所用的颜色：空间中的竖线全为红，水平线则有的蓝有的绿
            color_type = getLineColorType(view_idx, color_type)
        color = p_img.new_zeros(3)
        color[color_type] = 255
        if type.find("pk") != -1:
            color = torch.tensor([255, 255, 255], dtype=torch.float, device=p_img.device)
            prob = output["p_preds_xy"].new_tensor(
                PerspectiveDataset.generate_gradual_hough_label(input["peaks"][img_idx][view_idx][0],
                                                                input["xLabels"].shape[2], type="nearest_k", base=0.5))
        elif type.find("gt") != -1:
            prob = input["xLabels"][img_idx, view_idx]
        if prob is not None:  # 特殊情况，根据prob实时计算mat
            mat = prob.unsqueeze(0).expand(p_img.shape[1], -1)
        else:  # 利用先前在后处理步骤中算完的
            generatePred2DMask(cfg, input, output, img_idx)
            mat = output["p_preds_2dmask"][img_idx][view_idx, 0]
    elif type.find("y") == 0:
        color_type = 1
        if type.find("cl2") != -1:
            color_type = getLineColorType(view_idx, color_type)
        color = p_img.new_zeros(3)
        color[color_type] = 255
        if type.find("pk") != -1:
            color = torch.tensor([255, 255, 255], dtype=torch.float, device=p_img.device)
            prob = output["p_preds_xy"].new_tensor(
                PerspectiveDataset.generate_gradual_hough_label(input["peaks"][img_idx][view_idx][1],
                                                                input["yLabels"].shape[2], type="nearest_k", base=0.5))
        elif type.find("gt") != -1:
            prob = input["yLabels"][img_idx, view_idx]
        if prob is not None:  # 特殊情况，根据prob实时计算mat
            mat = prob.unsqueeze(1).expand(-1, p_img.shape[2])
        else:  # 利用先前在后处理步骤中算完的
            generatePred2DMask(cfg, input, output, img_idx)
            mat = output["p_preds_2dmask"][img_idx][view_idx, 1]
    elif type.find("c") == 0:
        def _genProb(probs):
            """
            把长为(angle_num,2)的，最后一维依次表示cup和cdown的霍夫域上的向量，拼接起来变为，cdown接在cup上的向量
            """
            # return torch.cat([probs[:, i] for i in range(probs.shape[1])]) # 该形式是原始的定义，与下面的完全等价但更复杂
            return probs.T.reshape(-1)

        color_type = 2
        if type.find("cl2") != -1:
            color_type = getLineColorType(view_idx, color_type)
        color = p_img.new_zeros(3)
        color[color_type] = 255
        if type.find("pk") != -1:
            color = torch.tensor([255, 255, 255], dtype=torch.float, device=p_img.device)
            prob = torch.cat([
                output["p_preds_cud"].new_tensor(
                    PerspectiveDataset.generate_gradual_hough_label(input["peaks"][img_idx][view_idx][2],
                                                                    input["cUpLabels"].shape[2],
                                                                    type="nearest_k", base=0.5)),
                output["p_preds_cud"].new_tensor(
                    PerspectiveDataset.generate_gradual_hough_label(input["peaks"][img_idx][view_idx][3],
                                                                    input["cDownLabels"].shape[2],
                                                                    type="nearest_k", base=0.5))])
        elif type.find("gt") != -1:
            prob = torch.cat(
                [input["cUpLabels"][img_idx, view_idx], input["cDownLabels"][img_idx, view_idx]])
        elif type.find("raw") != -1:
            prob = _genProb(output["raw_cud"][img_idx, view_idx])

        if prob is not None:  # 特殊情况，根据prob实时计算mat
            vote_mask_c_up_down = get_vote_mask_c_up_down(cfg, p_img)
            # mat = (prob * vote_mask_c_up_down).max(-1).values # OLD
            mat = (prob * vote_mask_c_up_down).sum(-1) / vote_mask_c_up_down.sum(-1)
        else:  # 利用先前在后处理步骤中算完的
            generatePred2DMask(cfg, input, output, img_idx)
            mat = output["p_preds_2dmask"][img_idx][view_idx, 2]
    elif type == "gtlines":
        color = torch.tensor([255, 255, 255], dtype=torch.float, device=p_img.device)
        thickness = int(round(p_img.shape[1] * 0.004))
        mat = np.zeros(p_img.shape[1:])
        for line in input["lines"][img_idx][view_idx]:
            cv2.line(mat, torch.round(line[3:5]).to(torch.int64).numpy(),
                     torch.round(line[5:7]).to(torch.int64).numpy(), 1.0, thickness=thickness)
        mat = p_img.new_tensor(mat)
    else:
        return None

    return mat, color  # mat: (h,w)


def getLineColorType(view_idx, line_type):
    if VIEW_NAME[view_idx] == "F" or VIEW_NAME[view_idx] == "B":
        if line_type == 0:
            rgb = 0
        elif line_type == 1:
            rgb = 2
        else:
            rgb = 1
    elif VIEW_NAME[view_idx] == "L" or VIEW_NAME[view_idx] == "R":
        if line_type == 0:
            rgb = 0
        elif line_type == 1:
            rgb = 1
        else:
            rgb = 2
    elif VIEW_NAME[view_idx] == "U" or VIEW_NAME[view_idx] == "D":
        if line_type == 0:
            rgb = 1
        elif line_type == 1:
            rgb = 2
        else:
            rgb = 0
    return rgb


def getGTLines2DMasks(cfg, input, output, img_idx):
    masks2d = []
    for view_idx in range(6):
        p_img = input["p_imgs"][img_idx, view_idx]
        thickness = int(round(p_img.shape[1] * 0.01))
        mat = np.zeros(p_img.shape)
        for line in input["lines"][img_idx][view_idx]:
            color_type = getLineColorType(view_idx, line[7])
            cv2.line(mat[color_type], torch.round(line[3:5]).to(torch.int64).numpy(),
                     torch.round(line[5:7]).to(torch.int64).numpy(), 1.0, thickness=thickness)
        masks2d.append(input["p_imgs"].new_tensor(mat))
    masks2d = torch.stack(masks2d)

    maskEq = _cal_p_pred_emask(cfg, masks2d, input["p_imgs"].shape[-2:], input["e_img"].shape[-2:])
    return masks2d, maskEq


def cvtRGBMatToDrawingNdArray(input):
    """
    :param input tensor(3,h,w) float型 范围0~1
    :return ndarray(h,w,4) int型 范围0~255
    """
    input4 = torch.cat([input, torch.clamp(input.max(0)[0], 0.0, 1.0).unsqueeze(0)], 0)
    return torch.clamp(torch.round(input4 * 255), 0.0, 255.0).permute(1, 2, 0).cpu().to(torch.uint8).numpy()


def drawEqualRectCorners(cfg, ax, type, input, output, img_idx, gt_cor_id, pred_cor_id):
    e_img = input["e_img"][img_idx].permute(1, 2, 0).cpu().numpy()
    ax.imshow(e_img)
    if type.find("r") != -1:
        cor = gt_cor_id.cpu().numpy()
        ax.scatter(cor[:, 0], cor[:, 1], c="red", s=10)
        if pred_cor_id is not None:
            cor = pred_cor_id.cpu().numpy()
            ax.scatter(cor[:, 0], cor[:, 1], c="green", s=10)
    if type.find("m") != -1:
        for one_draw_idx in range(3):
            generatePred2DMask(cfg, input, output, img_idx)
            mat = output["p_preds_emask"][img_idx][one_draw_idx]
            color = mat.new_zeros(3)
            color[one_draw_idx] = 255
            mask_img = torch.cat([color.repeat(*mat.shape[0:2], 1), mat.unsqueeze(-1) * 255], 2)
            mask_img = torch.round(mask_img).to(torch.uint8)
            ax.imshow(mask_img.cpu().numpy())
    if type.find("gt") != -1:
        _, emask = getGTLines2DMasks(cfg, input, output, img_idx)
        drawArray = cvtRGBMatToDrawingNdArray(emask)
        ax.imshow(drawArray)
    if type.find("w") != -1:
        drawWireframeOnEImg(ax, e_img, gt_cor_id, (0.0, 1.0, 0.0))
        drawWireframeOnEImg(ax, e_img, pred_cor_id, (1.0, 0.0, 0.0))


def o3dRunVis(vis):
    vis.update_geometry()
    vis.update_renderer()
    vis.poll_events()
    vis.run()


def o3dDrawLines(vis, lines_results, lwh, color=None):
    points, lines, colors = cvtLinesResultsForDraw(lines_results, lwh, color)
    line_pcd = o3d.LineSet()
    line_pcd.lines = o3d.Vector2iVector(lines)
    line_pcd.colors = o3d.Vector3dVector(colors)
    line_pcd.points = o3d.Vector3dVector(points)
    vis.add_geometry(line_pcd)


def pyplotGetCameraPos(gt_lwh):
    return (2 * gt_lwh[2] - gt_lwh[3]).item()


def pyplotDrawLines(ax, cameraPos, lines_results, lwh, color=None):
    points, lines, colors = cvtLinesResultsForDraw(lines_results, lwh, color)
    points2d = points - np.array([0.0, cameraPos, 0.0])
    points2d /= points2d[:, 1:2]
    points2d = points2d[:, [0, 2]]

    ax.set_facecolor("black")
    for i, line in enumerate(lines):
        p = points2d[line]
        ax.plot(p[:, 0], p[:, 1], c=colors[i] if colors is not None else None, linewidth=1)


def drawWireframeOnEImg(ax, e_img, cor, color):
    from visualization_from_json import wireframeGetMaskImg
    ax.imshow(wireframeGetMaskImg(e_img, cor, color).cpu().numpy())


def cvtLinesResultsForDraw(lines_results, lwh, color=None):
    points = []
    lines = []
    colors = []
    for line in lines_results:
        if color is None:
            # if 0 <= line[6] <= 1:
            #     color = [1.0, 0.0, 0.0]
            # elif 2 <= line[6] <= 3:
            #     color = [0.0, 1.0, 0.0]
            # elif 4 <= line[6] <= 5:
            #     color = [0.0, 0.0, 1.0]
            # elif 6 <= line[6] <= 7:
            #     color = [0.0, 0.0, 1.0]
            if line[2] == 1:
                color = [0.0, 1.0, 1.0]
            else:
                color = [0.0, 1.0, 0.0]
        if line[1] == 0:
            points.append([lwh[0], line[4], line[5]])
            points.append([lwh[1], line[4], line[5]])
        elif line[1] == 1:
            points.append([line[3], lwh[2], line[5]])
            points.append([line[3], lwh[3], line[5]])
        elif line[1] == 2:
            points.append([line[3], line[4], lwh[4]])
            points.append([line[3], line[4], lwh[5]])
        else:
            assert False
        lines.append([len(points) - 2, len(points) - 1])
        colors.append(color)
    return np.array(points), np.array(lines), colors


def o3dInitVis():
    """
    Open3D自带的坐标轴中，红色是x轴，绿色是y轴，蓝色是z轴！
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()

    def save_view_point(vis, filename):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(filename, param)

    vis.register_key_callback(ord("S"), lambda vis: save_view_point(vis, "./view-" + str(int(time.time())) + ".json"))
    vis.create_window(width=1386, height=752)
    if os.path.exists("./view-1617968465.json"):
        vis.get_view_control().convert_from_pinhole_camera_parameters(
            o3d.read_pinhole_camera_parameters("./view-1617968465.json"))
    renderOption: o3d.RenderOption = vis.get_render_option()
    renderOption.background_color = np.array([0, 0, 0], dtype=np.float32)
    renderOption.show_coordinate_frame = True
    renderOption.point_size = 0.1
    return vis


def makeLwhLines(lwh):
    result = [
        [0, 0, 0, math.nan, lwh[2], lwh[4], 0, 0],
        [0, 0, 0, math.nan, lwh[2], lwh[5], 0, 0],
        [0, 0, 0, math.nan, lwh[3], lwh[4], 0, 0],
        [0, 0, 0, math.nan, lwh[3], lwh[5], 0, 0],
        [0, 1, 0, lwh[0], math.nan, lwh[4], 0, 0],
        [0, 1, 0, lwh[0], math.nan, lwh[5], 0, 0],
        [0, 1, 0, lwh[1], math.nan, lwh[4], 0, 0],
        [0, 1, 0, lwh[1], math.nan, lwh[5], 0, 0],
        [0, 2, 0, lwh[0], lwh[2], math.nan, 0, 0],
        [0, 2, 0, lwh[0], lwh[3], math.nan, 0, 0],
        [0, 2, 0, lwh[1], lwh[2], math.nan, 0, 0],
        [0, 2, 0, lwh[1], lwh[3], math.nan, 0, 0],
    ]
    return lwh.new_tensor(result)


def visualize(cfg, input, output, drawtypes=None, show=False, dpi=None) -> Dict[str, np.ndarray]:
    postResults = [postProcess(cfg, input, output, img_idx) for img_idx in range(input["p_imgs"].shape[0])]
    return visualizeWithPostResults(cfg, input, output, postResults, drawtypes, show, dpi)


DRAW_3D_PREDBOX_COLOR = "blue"
DRAW_3D_GTBOX_COLOR = "white"


def visualizeWithPostResults(cfg, input, output, postResults: list, drawtypes=None, show=False, dpi=None) -> Dict[
    str, np.ndarray]:
    """
    结果可视化
    :param input 数据集给出的输入
    :param output 模型给出的输出
    :param postResults 数组，内含input中的每张图片调用postProcess函数返回的结果
    :param post_result postProcess函数返回的值
    :return 字典，key是字符串，value是(h,w,3)的ndarray，图片的RGB矩阵。
    """
    if drawtypes is None:
        drawtypes = DEFAULT_DRAWTYPE
    result = {}

    with torch.no_grad():
        for img_idx in range(input["p_imgs"].shape[0]):
            (gt_lines, gt_lwh, gt_cor_id), (pred_lines, pred_lwh, pred_cor_id), metric = postResults[img_idx]
            for draw_idx, one_fig_types in enumerate(drawtypes):
                fig: Figure = plt.figure(dpi=dpi)
                fig.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)  # 清除四周边距

                if isinstance(one_fig_types, str) and one_fig_types.find("e") == 0:
                    # 画单一的全景图
                    ax = plt.gca()
                    clearAxesLines(ax)
                    drawEqualRectCorners(cfg, ax, one_fig_types, input, output, img_idx, gt_cor_id, pred_cor_id)
                elif isinstance(one_fig_types, str) and one_fig_types.find("3d") == 0:
                    ax = plt.gca()
                    clearAxesLines(ax)
                    cameraPos = pyplotGetCameraPos(gt_lwh)
                    if one_fig_types.find("predbox") != -1:
                        pyplotDrawLines(ax, cameraPos, makeLwhLines(pred_lwh), pred_lwh, DRAW_3D_PREDBOX_COLOR)
                    if one_fig_types.find("gtbox") != -1:
                        pyplotDrawLines(ax, cameraPos, makeLwhLines(gt_lwh), gt_lwh, DRAW_3D_GTBOX_COLOR)
                    pyplotDrawLines(ax, cameraPos, gt_lines, gt_lwh, "red")
                    pyplotDrawLines(ax, cameraPos, pred_lines, pred_lwh, "green")
                elif isinstance(one_fig_types, list):
                    # 画左上角、右下角、右上角
                    for type in one_fig_types:
                        if type.find("e") == 0:
                            ax = getCubeAxes(fig, "E")
                            drawEqualRectCorners(cfg, ax, type, input, output, img_idx, gt_cor_id, pred_cor_id)
                        elif type.find("3d") == 0 and gt_lwh is not None and pred_lwh is not None:
                            ax = getCubeAxes(fig, "3D")
                            cameraPos = pyplotGetCameraPos(gt_lwh)
                            if type.find("predbox") != -1:
                                pyplotDrawLines(ax, cameraPos, makeLwhLines(pred_lwh), pred_lwh, DRAW_3D_PREDBOX_COLOR)
                            if type.find("gtbox") != -1:
                                pyplotDrawLines(ax, cameraPos, makeLwhLines(gt_lwh), gt_lwh, DRAW_3D_GTBOX_COLOR)
                            pyplotDrawLines(ax, cameraPos, gt_lines, gt_lwh, "red")
                            if len(pred_lines) > 0:
                                pyplotDrawLines(ax, cameraPos, pred_lines, pred_lwh, "green")
                        elif type == "text":
                            ax = getCubeAxes(fig, "TEXT")
                            toWrite = ""
                            if "CE" in metric and "PE" in metric:
                                toWrite = "CE:{:.3f} PE:{:.3f}\n" \
                                          "3DIoU:{:.2f}\n" \
                                          " ".format(
                                    # "gt:{:s}\n   {:s}\n" \
                                    # "pr:{:s}\n   {:s}\n".format(
                                    metric["CE"], metric["PE"], metric["3DIoU"],
                                    ",".join(["{:.2f}".format(v.item()) for v in gt_lwh[0:3]]),
                                    ",".join(["{:.2f}".format(v.item()) for v in gt_lwh[3:6]]),
                                    ",".join(["{:.2f}".format(v.item()) for v in pred_lwh[0:3]]),
                                    ",".join(["{:.2f}".format(v.item()) for v in pred_lwh[3:6]]), )
                            elif "rmse" in metric and "delta_1" in metric:
                                toWrite = "3DIoU:{:.2f}\n" \
                                          "2DIoU:{:.2f}\n" \
                                          "delta_1:{:.3f}\n" \
                                          " ".format(
                                    # "rmse:{:.4f}\n".format(
                                    metric["3DIoU"], metric["2DIoU"], metric["delta_1"], metric["rmse"],
                                )
                            ax.text(0, 0, toWrite)
                            # if "nz" in metric:
                            #     ax.text(0, 0.85, "nz:" + metric["nz"], color="blue")
                            # if "noline" in metric:
                            #     ax.text(0, 0.75, "nl:" + metric["noline"], color="red")

                    mask_buffer = {}

                    gtlines_colored_masks2d = None

                    # 画每个面
                    for view_idx, view_name in enumerate(VIEW_NAME):
                        ax = getCubeAxes(fig, view_name)
                        p_img = input["p_imgs"][img_idx, view_idx]
                        if one_fig_types[0] != "canny":
                            ax.imshow(p_img.permute(1, 2, 0).cpu().numpy())
                            one_fig_types_2 = one_fig_types
                        else:
                            ax.imshow(np.expand_dims(output["canny_image"][img_idx][view_idx], 2).repeat(3, 2))
                            one_fig_types_2 = one_fig_types[1:]
                        mask_buffer[view_idx] = {}

                        # 画每种类型的线
                        for type in one_fig_types_2:
                            t = getMaskByType(type, cfg, input, output, img_idx, view_idx)
                            if t is not None:
                                mat, color = t
                                mask_buffer[view_idx][type] = mat

                                mask_img = torch.cat([color.repeat(*mat.shape[0:2], 1), mat.unsqueeze(-1) * 255], 2)
                                mask_img = torch.round(mask_img).to(torch.uint8)
                                ax.imshow(mask_img.cpu().numpy())

                            if type == "gtlines_colored":
                                if gtlines_colored_masks2d is None:
                                    gtlines_colored_masks2d, _ = getGTLines2DMasks(cfg, input, output, img_idx)
                                drawArray = cvtRGBMatToDrawingNdArray(gtlines_colored_masks2d[view_idx])
                                ax.imshow(drawArray)

                            if type == "border":
                                # 画一个白边框
                                BORDER_WIDTH = 2
                                img_hw = p_img.shape[-2:]
                                white_border_mask = np.ones((*img_hw, 4), dtype=np.uint8) * 255
                                white_border_mask[BORDER_WIDTH:img_hw[0] - BORDER_WIDTH,
                                BORDER_WIDTH:img_hw[1] - BORDER_WIDTH, 3] = 0
                                ax.imshow(white_border_mask)

                            if type.find("hough_line") == 0:
                                ax.set_xlim(0, 512)
                                ax.set_ylim(0, 512)
                                ax.invert_yaxis()
                                liness = output["hough_lines"][img_idx][view_idx]
                                for ii, lines in enumerate(liness):
                                    for jj, line in enumerate(lines):
                                        color = "b"
                                        if jj == 0:
                                            if type.find("red") != -1: continue
                                        else:
                                            if type.find("first_only") != -1:
                                                continue
                                        ax.plot([line[0], line[2]], [line[1], line[3]], color)

                                if type.find("red") != -1:
                                    for ii, lines in enumerate(liness):
                                        if len(lines) > 0:
                                            line = lines[0]
                                            color = "r"
                                            ax.plot([line[0], line[2]], [line[1], line[3]], color)

                # 获得图片
                buf = io.BytesIO()
                fig.savefig(buf, format="jpg")
                buf.seek(0)
                img = Image.open(buf)  # 使用Image打开图片数据
                img = np.asarray(img)
                buf.close()
                if show:
                    fig.show()
                else:
                    result["{:s}--{:s}".format(input["filename"][img_idx], str(draw_idx))] = img
                plt.close()
    return result
