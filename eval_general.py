import argparse
import json

import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm

from eval_cuboid import prepare_gtdt_pairs
from misc import post_proc, panostretch


def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first) * 2 - 1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys


def cor_2_1d(cor, H, W):
    bon_ceil_x, bon_ceil_y = [], []
    bon_floor_x, bon_floor_y = [], []
    n_cor = len(cor)
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(cor[i * 2],
                                              cor[(i * 2 + 2) % n_cor],
                                              z=-50, w=W, h=H)
        bon_ceil_x.extend(xys[:, 0])
        bon_ceil_y.extend(xys[:, 1])
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(cor[i * 2 + 1],
                                              cor[(i * 2 + 3) % n_cor],
                                              z=50, w=W, h=H)
        bon_floor_x.extend(xys[:, 0])
        bon_floor_y.extend(xys[:, 1])
    bon_ceil_x, bon_ceil_y = sort_xy_filter_unique(bon_ceil_x, bon_ceil_y, y_small_first=True)
    bon_floor_x, bon_floor_y = sort_xy_filter_unique(bon_floor_x, bon_floor_y, y_small_first=False)
    bon = np.zeros((2, W))
    bon[0] = np.interp(np.arange(W), bon_ceil_x, bon_ceil_y, period=W)
    bon[1] = np.interp(np.arange(W), bon_floor_x, bon_floor_y, period=W)
    bon = ((bon + 0.5) / H - 0.5) * np.pi
    return bon


def layout_2_depth(cor_id, h, w, return_mask=False):
    # Convert corners to per-column boundary first
    # Up -pi/2,  Down pi/2
    vc, vf = cor_2_1d(cor_id, h, w)
    vc = vc[None, :]  # [1, w]
    vf = vf[None, :]  # [1, w]
    assert (vc > 0).sum() == 0
    assert (vf < 0).sum() == 0

    # Per-pixel v coordinate (vertical angle)
    vs = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
    vs = np.repeat(vs[:, None], w, axis=1)  # [h, w]

    # Floor-plane to depth
    floor_h = 1.6
    floor_d = np.abs(floor_h / np.sin(vs))

    # wall to camera distance on horizontal plane at cross camera center
    cs = floor_h / np.tan(vf)

    # Ceiling-plane to depth
    ceil_h = np.abs(cs * np.tan(vc))  # [1, w]
    ceil_d = np.abs(ceil_h / np.sin(vs))  # [h, w]

    # Wall to depth
    wall_d = np.abs(cs / np.cos(vs))  # [h, w]

    # Recover layout depth
    floor_mask = (vs > vf)
    ceil_mask = (vs < vc)
    wall_mask = (~floor_mask) & (~ceil_mask)
    depth = np.zeros([h, w], np.float32)  # [h, w]
    depth[floor_mask] = floor_d[floor_mask]
    depth[ceil_mask] = ceil_d[ceil_mask]
    depth[wall_mask] = wall_d[wall_mask]

    assert (depth == 0).sum() == 0
    if return_mask:
        return depth, floor_mask, ceil_mask, wall_mask
    return depth


def test_general(dt_cor_id, gt_cor_id, w, h, losses):
    dt_floor_coor = dt_cor_id[1::2]
    dt_ceil_coor = dt_cor_id[0::2]
    gt_floor_coor = gt_cor_id[1::2]
    gt_ceil_coor = gt_cor_id[0::2]
    assert (dt_floor_coor[:, 0] != dt_ceil_coor[:, 0]).sum() == 0
    assert (gt_floor_coor[:, 0] != gt_ceil_coor[:, 0]).sum() == 0

    # Eval 3d IoU and height error(in meter)
    N = len(dt_floor_coor)
    ch = -1.6
    dt_floor_xy = post_proc.np_coor2xy(dt_floor_coor, ch, 1024, 512, floorW=1, floorH=1)
    gt_floor_xy = post_proc.np_coor2xy(gt_floor_coor, ch, 1024, 512, floorW=1, floorH=1)
    dt_poly = Polygon(dt_floor_xy)
    gt_poly = Polygon(gt_floor_xy)
    if not gt_poly.is_valid:
        print('Skip ground truth invalid (%s)' % gt_path)
        return

    # 2D IoU
    try:
        area_dt = dt_poly.area
        area_gt = gt_poly.area
        area_inter = dt_poly.intersection(gt_poly).area
        iou2d = area_inter / (area_gt + area_dt - area_inter)
    except:
        iou2d = 0.0

    # 3D IoU
    try:
        cch_dt = post_proc.get_z1(dt_floor_coor[:, 1], dt_ceil_coor[:, 1], ch, 512)
        cch_gt = post_proc.get_z1(gt_floor_coor[:, 1], gt_ceil_coor[:, 1], ch, 512)
        h_dt = abs(cch_dt.mean() - ch)
        h_gt = abs(cch_gt.mean() - ch)
        area3d_inter = area_inter * min(h_dt, h_gt)
        area3d_pred = area_dt * h_dt
        area3d_gt = area_gt * h_gt
        iou3d = area3d_inter / (area3d_pred + area3d_gt - area3d_inter)
    except:
        iou3d = 0.0

    # rmse & delta_1
    gt_layout_depth = layout_2_depth(gt_cor_id, h, w)
    try:
        dt_layout_depth = layout_2_depth(dt_cor_id, h, w)
    except:
        dt_layout_depth = np.zeros_like(gt_layout_depth)
    rmse = ((gt_layout_depth - dt_layout_depth) ** 2).mean() ** 0.5
    thres = np.maximum(gt_layout_depth / dt_layout_depth, dt_layout_depth / gt_layout_depth)
    delta_1 = (thres < 1.25).mean()

    # Add a result
    n_corners = len(gt_floor_coor)
    if n_corners % 2 == 1:
        n_corners_str = 'odd'
    elif n_corners < 10:
        n_corners_str = str(n_corners)
    else:
        n_corners_str = '10+'
    losses["n_corners_type"] = n_corners_str
    losses['2DIoU'] = iou2d * 100
    losses['3DIoU'] = iou3d * 100
    losses['rmse'] = rmse
    losses['delta_1'] = delta_1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dt_glob',
                        help='NOTE: Remeber to quote your glob path.'
                             'Files assumed to be json from inference.py')
    parser.add_argument('--gt_glob',
                        help='NOTE: Remeber to quote your glob path.'
                             'Files assumed to be txt')
    parser.add_argument('--w', default=1024, type=int,
                        help='GT images width')
    parser.add_argument('--h', default=512, type=int,
                        help='GT images height')
    args = parser.parse_args()

    # Prepare (gt, dt) pairs
    gtdt_pairs = prepare_gtdt_pairs(args.gt_glob, args.dt_glob)

    # Testing
    losses = dict([
        (n_corner, {'2DIoU': [], '3DIoU': [], 'rmse': [], 'delta_1': []})
        for n_corner in ['4', '6', '8', '10+', 'odd', 'overall']
    ])
    for gt_path, dt_path in tqdm(gtdt_pairs, desc='Testing'):
        # Parse ground truth
        with open(gt_path) as f:
            gt_cor_id = np.array([l.split() for l in f], np.float32)

        # Parse inferenced result
        with open(dt_path) as f:
            dt = json.load(f)
        dt_cor_id = np.array(dt['uv'], np.float32)
        dt_cor_id[:, 0] *= args.w
        dt_cor_id[:, 1] *= args.h

        test_general(dt_cor_id, gt_cor_id, args.w, args.h, losses)

    for k, result in losses.items():
        iou2d = np.array(result['2DIoU'])
        iou3d = np.array(result['3DIoU'])
        rmse = np.array(result['rmse'])
        delta_1 = np.array(result['delta_1'])
        if len(iou2d) == 0:
            continue
        print('GT #Corners: %s  (%d instances)' % (k, len(iou2d)))
        print('    2DIoU  : %.2f' % (iou2d.mean() * 100))
        print('    3DIoU  : %.2f' % (iou3d.mean() * 100))
        print('    RMSE   : %.2f' % (rmse.mean()))
        print('    delta^1: %.2f' % (delta_1.mean()))
