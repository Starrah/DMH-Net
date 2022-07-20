from functools import reduce

import numpy as np
import py360convert

from py360convert import rotation_matrix


def rotationMatrix(u, v, in_rot):
    Rx = rotation_matrix(v, [1, 0, 0])
    Ry = rotation_matrix(u, [0, 1, 0])
    Ri = rotation_matrix(in_rot, np.array([0, 0, 1.0]).dot(Rx).dot(Ry))
    return Rx.dot(Ry).dot(Ri)


def unitxyzToPerspectiveCoord(input, fov_deg, u_deg, v_deg, out_hw, in_rot_deg=0):
    """
    把unitxyz转化为单个perspective image上的xy像素坐标。
    :param input: (n, 3) n个点、每个点三维坐标(unitxyz)
    :param fov_deg: 同py360convert.e2p函数
    :param u_deg: 同py360convert.e2p函数
    :param v_deg: 同py360convert.e2p函数
    :param out_hw: 同py360convert.e2p函数
    :param in_rot_deg: 同py360convert.e2p函数
    :return: 元组。第一个元素是(n, 2) n个点、每个点在图像上的两维坐标（以像素为单位）；第二个元素是(n, 3)，每个点在透视坐标系中的坐标
    """
    assert len(input.shape) == 2 and input.shape[1] == 3

    h_fov, v_fov = fov_deg[0] * np.pi / 180, fov_deg[1] * np.pi / 180
    in_rot = in_rot_deg * np.pi / 180
    u = -u_deg * np.pi / 180
    v = v_deg * np.pi / 180

    rotMat = rotationMatrix(u, v, in_rot)
    imgXyz = input.dot(rotMat.T)  # 对于旋转矩阵，其逆等于其转置
    imgXyz = imgXyz / np.abs(imgXyz[:, 2:])  # 使得z变为1

    # 根据x y和fov，计算出可视化区域的range
    x_max = np.tan(h_fov / 2)
    y_max = np.tan(v_fov / 2)
    # 将像素范围和range范围建立线性对应
    normed_pos = imgXyz[:, :2] / np.array([x_max, -y_max], dtype=imgXyz.dtype) / 2 + 0.5
    pos = normed_pos * np.array([out_hw[1] - 1, out_hw[0] - 1], dtype=imgXyz.dtype)

    return pos, imgXyz


def coordE2P(input_pts, img, fov_deg, u_deg, v_deg, out_hw, in_rot_deg=0):
    """
    把equirect下的坐标转换为perspective下的坐标
    :param input: (n, 3) n个点、每个点三维坐标(unitxyz)
    :param img: 全景equirect图像
    :param fov_deg: 同py360convert.e2p函数
    :param u_deg: 同py360convert.e2p函数
    :param v_deg: 同py360convert.e2p函数
    :param out_hw: 同py360convert.e2p函数
    :param in_rot_deg: 同py360convert.e2p函数
    :return: 元组。第一个元素是(n, 2) n个点、每个点两维坐标；第二个元素是整数类型(n) 表示每个点的类型：
    0表示点不在180度视角内（在相机后面），xy值无意义；1表示点在180度视角内、但不在图片内，2表示点在图片内。
    """
    uv = py360convert.coor2uv(input_pts, img.shape[0], img.shape[1])
    xyz = py360convert.uv2unitxyz(uv)
    result, imgXyz = unitxyzToPerspectiveCoord(xyz, fov_deg, u_deg, v_deg, out_hw, in_rot_deg)
    type = (imgXyz[:, 2] > 0).astype(np.int8)
    inimage_mask = reduce(np.logical_and, [type, result[:, 0] >= 0, result[:, 0] <= out_hw[1] - 1, result[:, 1] >= 0,
                                           result[:, 1] <= out_hw[0] - 1])
    type[inimage_mask] = 2
    return result, type, imgXyz
