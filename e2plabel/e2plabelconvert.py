from typing import List, Dict

import numpy as np
import py360convert

from .convertExUtils import coordE2P

# 视图顺序： F R B L U D
#     -----
#     | 4 |
#     -----
# | 3 | 0 | 1 | 2 |
#     -----
#     | 5 |
#     -----
VIEW_ARGS = [
    [(90, 90), 0, 0],
    [(90, 90), 90, 0],
    [(90, 90), 180, 0],
    [(90, 90), -90, 0],
    [(90, 90), 0, 90],
    [(90, 90), 0, -90],
]
VIEW_NAME = ['F', 'R', 'B', 'L', 'U', 'D']
VIEW_SIZE = (512, 512)  # perspective image的size


def generatePerspective(e_img: np.ndarray, cor: np.ndarray, view_name=VIEW_NAME, view_args=VIEW_ARGS,
                        view_size=VIEW_SIZE) -> List[Dict[str, np.ndarray]]:
    """
    :param e_img 全景图片
    :param cor 角点在全景图片上的坐标
    :return 含有六个元素的数组，每个元素是一个字典，表示一个面；字典内含 {
        p_img:图像,
        name: 面的名称,
        points:(n,2)，表示点在图片中的x、y坐标（浮点数）
        point_types:整数(n)，对应于points中每个点表示其类型，0表示点不在180度视角内（在相机后面），xy值无意义；1表示点在180度视角内、但不在图片内，2表示点在图片内。
        lines:(k,7)，表示图中能看到的k条线。每条线用七个数表示，前两个是端点在points中的序号，然后是线的类型：0是竖直的墙壁线，1是天花板线，2是地板线，然后是起点的x、y坐标，然后是终点的x、y坐标
    }
    """
    result = []
    for view_idx, (view_name, view) in enumerate(zip(view_name, view_args)):
        p_img = py360convert.e2p(e_img, *view, view_size)
        r = generateOnePerspectiveLabel(e_img, cor, *view, VIEW_SIZE)
        r["name"] = view_name
        r["img"] = p_img
        result.append(r)
    return result


def linesPostProcess(lines, img_hw, is_updown_view, return_mask=False):
    """
    对线进行处理，筛选掉看不见的线、对线的起终点进行规范化处理
    :param lines:(k,7)，表示图中的k条线。每条线用七个数表示，前两个是端点在points中的序号，然后是线的类型：0是竖直的墙壁线，1是天花板线，2是地板线，然后是起点的x、y坐标，然后是终点的x、y坐标
    :param img_hw:(2)，图片的宽和高
    :return (m,8)图中能看到的m条线。每条线前7个数含义同上，第8个数表示线在视图中的方向：0竖直线1水平线2过中心线
    """

    # ！！！根据10月12日推导出的结果：使用py360convert的e2p、e2c变换得出的点，3d空间中坐标-1~1对应的范围应当是2d图片中像素0~h-1，而不是0~h！
    # 以h=512为例，在图片2d坐标系下的511.5是没有意义（在当前平面内不可见）的！
    # 因此作出修改！
    # 虽然分析觉得，这个小错误并不会对结果有实质性的影响（因为最多只会差一个像素），但还是改过来吧！
    # unitxyzToPerspectiveCoord、coordE2P、generateOnePerspectiveLabel、lineCoordToRatio四个函数也做了相同的修改。

    img_hw = [img_hw[0] - 1, img_hw[1] - 1]

    def processPoint(point, k):
        xInRange = False
        if point[0] < 0:
            y = k * (0 - point[0]) + point[1]
            if 0 <= y <= img_hw[0]:
                return 0, y
        elif point[0] > img_hw[1]:
            y = k * (img_hw[1] - point[0]) + point[1]
            if 0 <= y <= img_hw[0]:
                return img_hw[1], y
        else:
            xInRange = True
        if point[1] < 0:
            x = (0 - point[1]) / k + point[0]
            if 0 <= x <= img_hw[1]:
                return x, 0
        elif point[1] > img_hw[0]:
            x = (img_hw[0] - point[1]) / k + point[0]
            if 0 <= x <= img_hw[1]:
                return x, img_hw[0]
        else:
            if xInRange:
                return point
        return None

    result = []
    mask = []
    for line in lines:
        k = (line[6] - line[4]) / (line[5] - line[3])
        p1Res = processPoint(line[3:5], k)
        p2Res = processPoint(line[5:7], k)
        if p1Res is not None and p2Res is not None:
            if line[2] == 0:
                direct = 2 if is_updown_view else 0
            else:
                if is_updown_view:
                    direct = 1 if -1 <= k <= 1 else 0
                else:
                    yLR = (k * (0 - p1Res[0]) + p1Res[1], k * (img_hw[1] - p1Res[0]) + p1Res[1])
                    if (yLR[0] < img_hw[0] / 2 and yLR[1] > img_hw[0] / 2) or (
                            yLR[1] < img_hw[0] / 2 and yLR[0] > img_hw[0] / 2):
                        direct = 2
                    else:
                        direct = 1
            result.append(np.concatenate((line[0:3], p1Res, p2Res, (direct,))))
            mask.append(True)
        else:
            mask.append(False)
    if not return_mask:
        return result
    else:
        return result, mask


def generateOnePerspectiveLabel(e_img, e_label, fov_deg, u_deg, v_deg, out_hw,
                                in_rot_deg=0):
    """
    根据给定的perspective参数，生成label信息，并可以可视化。
    :param e_img: 全景equirect图像
    :param e_label: (n, 2)在全景equirect坐标系下的gt角点坐标
    :param ax: 画图的pyplot.Axes
    # :param img_save_path: 若传入True，则把结果图像通过plt.show显示；若传入其他字符串，则保存为文件；否则不显示和保存结果图像。
    :param fov_deg: 同py360convert.e2p函数
    :param u_deg: 同py360convert.e2p函数
    :param v_deg: 同py360convert.e2p函数
    :param out_hw: 同py360convert.e2p函数
    :param in_rot_deg: 同py360convert.e2p函数
    :return: 字典，是图片中的点和线段信息，内含"points" "point_types" "lines" "lines" 三个字段。
    points:(n,2)，表示点在图片中的x、y坐标（浮点数）
    point_types:整数(n)，对应于points中每个点表示其类型，0表示点不在180度视角内（在相机后面），xy值无意义；1表示点在180度视角内、但不在图片内，2表示点在图片内。
    lines:(k,8)，表示图中的k条线。每条线用七个数表示，前两个是端点在points中的序号，然后是线的类型：0是竖直的墙壁线，1是天花板线，2是地板线，然后是起点的x、y坐标，然后是终点的x、y坐标，然后是线在视图中的类型-0竖直线1水平线2过中心线
    """
    points, point_types, imgXyz = coordE2P(e_label, e_img, fov_deg, u_deg, v_deg, out_hw, in_rot_deg)
    lines = []

    corner_count = e_label.shape[0]
    # 定义三种交线。内含12个数组代表长方体的12条边。
    # 每个数组的前两个元素对应着label中按顺序给定的点的序号。
    # 第三个元素表示这条线的类别。0是竖直的墙壁线，1是天花板线，2是地板线。
    LINES = []
    for i in range(0, corner_count, 2):
        LINES.append([i, i + 1, 0])
        LINES.append([i, (i + 2) % corner_count, 1])
        LINES.append([i + 1, (i + 3) % corner_count, 2])

    for l in LINES:
        if point_types[l[0]] > 0 and point_types[l[1]] > 0:
            # 两个点都在相机前方180度视角范围内
            lines.append(np.concatenate([l, points[l[0]], points[l[1]]]))
        elif point_types[l[0]] == 2 or point_types[l[1]] == 2:
            # 只有一个点在图像内、另一个点在相机后方
            # 从连成的直线上找合适的点，这个点投影位于边线上
            line = np.concatenate([l, points[l[0]], points[l[1]]])
            if point_types[l[0]] == 2:
                p1 = imgXyz[l[0]]
                p2 = imgXyz[l[1]]
                toFill = 1
            else:
                p1 = imgXyz[l[1]]
                p2 = imgXyz[l[0]]
                toFill = 0
            tantheta = np.tan((180 - fov_deg[0]) / 2 * np.pi / 180)
            # pc：p1和p2连线，与FOV平面的交点。k应该能保证在0.5~1之间。pc的x应该能保证非常接近边缘（1或-1）？
            k1 = (p2[0] * tantheta - p2[2]) / ((p1[2] - p2[2]) - ((p1[0] - p2[0]) * tantheta))
            k2 = (p2[0] * -tantheta - p2[2]) / ((p1[2] - p2[2]) - ((p1[0] - p2[0]) * -tantheta))
            k = k1 if 0.5 <= k1 <= 1 else k2
            pc = p1 * k + p2 * (1 - k)
            assert 0.5 <= k <= 1 and pc[2] > 0, "k error"
            pc = pc / pc[2]
            assert -0.01 <= (abs(pc[0]) - 1) <= 0.01, "pc error"
            # 把pc的坐标，往图像上映射
            h_fov, v_fov = fov_deg[0] * np.pi / 180, fov_deg[1] * np.pi / 180
            x_max = np.tan(h_fov / 2)
            y_max = np.tan(v_fov / 2)
            normed_pos = pc[:2] / np.array([x_max, -y_max], dtype=imgXyz.dtype) / 2 + 0.5
            pos = normed_pos * np.array([out_hw[1] - 1, out_hw[0] - 1], dtype=imgXyz.dtype)
            line[2 * toFill + 3:2 * toFill + 5] = pos  # 把pc点的图像坐标填入
            lines.append(line)

    lines = linesPostProcess(lines, out_hw, v_deg != 0)  # 去除看不见的线
    result = {
        "points": points,
        "point_types": point_types,
        "lines": lines
    }
    return result
