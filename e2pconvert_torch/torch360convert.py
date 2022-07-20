import numpy as np
import torch


def coor2uv(coorxy, h, w):
    coor_x, coor_y = coorxy[:, 0:1], coorxy[:, 1:2]
    u = ((coor_x + 0.5) / w - 0.5) * 2 * np.pi
    v = -((coor_y + 0.5) / h - 0.5) * np.pi

    return torch.cat([u, v], -1)


def uv2unitxyz(uv):
    u, v = uv[:, 0:1], uv[:, 1:2]
    y = torch.sin(v)
    c = torch.cos(v)
    x = c * torch.sin(u)
    z = c * torch.cos(u)

    return torch.cat([x, y, z], -1)


def uv2coor(uv, h, w):
    '''
    uv: ndarray in shape of [..., 2]
    h: int, height of the equirectangular image
    w: int, width of the equirectangular image
    '''
    u, v = uv[:, 0:1], uv[:, 1:2]
    coor_x = (u / (2 * np.pi) + 0.5) * w - 0.5
    coor_y = (-v / np.pi + 0.5) * h - 0.5

    return torch.cat([coor_x, coor_y], -1)


def xyz2uv(xyz):
    '''
    xyz: ndarray in shape of [..., 3]
    '''
    x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
    u = torch.atan2(x, z)
    c = torch.sqrt(x ** 2 + z ** 2)
    v = torch.atan2(y, c)

    return torch.cat([u, v], -1)
