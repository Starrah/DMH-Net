import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from scipy.ndimage.filters import maximum_filter
from scipy.spatial import distance

PI = float(np.pi)


def interp_point_by_u(xy0, xy1, u0, u1, u):
    c0 = np.linalg.norm(xy0)
    c1 = np.linalg.norm(xy1)
    ratio = np.abs(np.sin(u - u1) * c1 / np.sin(u - u0) / c0)
    return xy0 + (xy1 - xy0) * 1 / (1 + ratio)


def get_maxi_loc(signal, min_v, r):
    if (len(signal) == 0):
        return np.empty(0, )
    if np.mean(signal) > min_v:
        min_v = np.mean(signal)
    max_v = maximum_filter(signal, size=r, mode='wrap')
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    #print(pk_loc.shape)
    return pk_loc


'''
def gen_doors(cor_id, y_seg_, r, min_v, cam_height = 1.6, W = 1024, H = 512):
    ww_loc = cor_id[::2,0].astype(int)
    floor_z = -cam_height
    floor_xy = np_coor2xy(cor_id[1::2], floor_z, W, H, floorW=1, floorH=1)
    
    u = np_coorx2u(cor_id[1::2, 0], W)
    signal_full = y_seg_
    doors_on_wall = []
    doors_score = []
    
    for idx in range(0, ww_loc.shape[0]):
        start_xy = floor_xy[idx, :]
        end_xy = floor_xy[(idx+1) % ww_loc.shape[0], :]
        if(np.sum(abs(start_xy - end_xy)) < 0.6):
            continue
        if(idx == ww_loc.shape[0] - 1):
            signal1 = signal_full[np.arange(ww_loc[-1],W)]
            signal2 = signal_full[np.arange(0, ww_loc[0])]
            pk_loc1 = get_maxi_loc(signal1, min_v, r)
            pk_loc1 += ww_loc[-1]
            pk_loc2 = get_maxi_loc(signal2, min_v, r)
            pk_loc = np.concatenate([pk_loc1,pk_loc2])
            #print('pk', pk_loc1, pk_loc2)
        else:
            signal = signal_full[np.arange(ww_loc[idx], ww_loc[idx + 1])]
            pk_loc = get_maxi_loc(signal, min_v, r)
            pk_loc += ww_loc[idx]

        #max_v = max_v[pk_loc]
        #if(idx == ww_loc.shape[0] - 1):
        #    pk_loc = pk_loc % W
        u_doors = np_coorx2u(pk_loc, W)
        door_xys = []
        for u_door in u_doors:
            door_xy = interp_point_by_u(start_xy, end_xy, u[idx], u[(idx+1) % ww_loc.shape[0]], u_door)
            door_xys.append(door_xy)
        door_xys = np.array(door_xys)
        doors_on_wall.append(door_xys)
    return doors_on_wall
'''


def gen_doors(cor_id, xs_seg_wd_, xs_seg_dw_, cam_height=1.6, W=1024, H=512):
    ww_loc = cor_id[::2, 0].astype(int)
    floor_z = -cam_height
    floor_xy = np_coor2xy(cor_id[1::2], floor_z, W, H, floorW=1, floorH=1)

    u = np_coorx2u(cor_id[1::2, 0], W)
    cor = np.roll(ww_loc, 1, axis=0)

    wd_bounds_left = np.sign(xs_seg_wd_.reshape(-1, 1) - ww_loc.reshape(1, -1))
    wd_bounds_right = np.sign(xs_seg_wd_.reshape(-1, 1) - np.roll(ww_loc, -1, axis=0).reshape(1, -1))

    dw_bounds_left = np.sign(xs_seg_wd_.reshape(-1, 1) - ww_loc.reshape(1, -1))
    dw_bounds_right = np.sign(xs_seg_wd_.reshape(-1, 1) - np.roll(ww_loc, -1, axis=0).reshape(1, -1))

    doors_on_wall = []
    doors_score = []
    u_doors_lb = np_coorx2u(xs_seg_wd_, W)
    u_doors_rb = np_coorx2u(xs_seg_dw_, W)
    # print(u_doors_lb)
    for idx in range(0, ww_loc.shape[0]):
        start_xy = floor_xy[idx, :]
        end_xy = floor_xy[(idx + 1) % ww_loc.shape[0], :]
        start_u = ww_loc[idx]
        end_u = ww_loc[(idx + 1) % ww_loc.shape[0]]

        if (np.sum(abs(start_xy - end_xy)) < 0.6):
            continue
        if (idx == ww_loc.shape[0] - 1):
            signal1 = signal_full[np.arange(ww_loc[-1], W)]
            signal2 = signal_full[np.arange(0, ww_loc[0])]
            pk_loc1 = get_maxi_loc(signal1, min_v, r)
            pk_loc1 += ww_loc[-1]
            pk_loc2 = get_maxi_loc(signal2, min_v, r)
            pk_loc = np.concatenate([pk_loc1, pk_loc2])
            #print('pk', pk_loc1, pk_loc2)
        else:
            signal = signal_full[np.arange(ww_loc[idx], ww_loc[idx + 1])]
            pk_loc = get_maxi_loc(signal, min_v, r)
            pk_loc += ww_loc[idx]

        #max_v = max_v[pk_loc]
        #if(idx == ww_loc.shape[0] - 1):
        #    pk_loc = pk_loc % W
        u_doors = np_coorx2u(pk_loc, W)
        door_xys = []
        for u_door in u_doors:
            door_xy = interp_point_by_u(start_xy, end_xy, u[idx], u[(idx + 1) % ww_loc.shape[0]], u_door)
            door_xys.append(door_xy)
        door_xys = np.array(door_xys)
        doors_on_wall.append(door_xys)
    draw_floor_plan(floor_xy)
    return doors_on_wall


def check_even_dps(dps, del_candidates, MIN_DOOR_WIDTH=0.4, MAX_D2D_DIST=0.2):
    door_widths = np.linalg.norm(dps[::2, :] - dps[1::2, :], axis=1)
    if (np.sum(door_widths < MIN_DOOR_WIDTH) == 0):
        return dps
    # we can only solve this case by assuming head and tail are bad dets
    if (dps.shape[0] == 2):
        return
    shrink_door_widths = np.linalg.norm(dps[2::2, :] - dps[1:-1:2, :], axis=1)
    if (np.sum(shrink_door_widths < MIN_DOOR_WIDTH) == 0):
        if ((del_candidates[0] < MAX_D2D_DIST) and (del_candidates[1] < MAX_D2D_DIST)):
            dps = np.delete(dps, [0, -1], 0)
            return dps
    #fall back to whatever
    return dps


def check_odd_dps(dps, del_candidate_pair, MIN_DOOR_WIDTH=0.4, MAX_D2D_DIST=0.2):
    if (dps.shape[0] == 1):
        return
    door_widths_0 = np.linalg.norm(dps[0:-1:2, :] - dps[1::2, :], axis=1)
    door_widths_1 = np.linalg.norm(dps[1::2, :] - dps[2::2, :], axis=1)
    if ((del_candidate_pair[0] < MAX_D2D_DIST) and (np.sum(door_widths_1 < MIN_DOOR_WIDTH) == 0)):
        dps = np.delete(dps, 0, 0)
        return dps
    if ((del_candidate_pair[1] < MAX_D2D_DIST) and (np.sum(door_widths_0 < MIN_DOOR_WIDTH) == 0)):
        dps = np.delete(dps, -1, 0)
        return dps
    break_idx = np.argmin(np.linalg.norm(dps[0:-1:1, :] - dps[1::1, :],
                                         axis=1))
    dps = np.delete(dps, break_idx + 1, 0)
    door_widths = np.linalg.norm(dps[0:-1:2, :] - dps[1::2, :], axis=1)
    if (np.sum(door_widths_0 < MIN_DOOR_WIDTH) == 0):
        return dps
    return


def remove_narrow_doors(dps, MIN_DOOR_WIDTH):
    rows_to_del = []
    while True:  # can be optimized by only updating the width needed for updating
        dps = np.delete(dps, rows_to_del, 0)
        rows_to_del = []
        for dp_idx in range(len(dps)):
            cand_widths = [-1, -1]
            if (dp_idx > 0):
                cand_widths[0] = np.sum(np.abs(dps[dp_idx - 1] - dps[dp_idx]))
            if (dp_idx < len(dps) - 1):
                cand_widths[1] = np.sum(np.abs(dps[dp_idx + 1] - dps[dp_idx]))
            if (np.array(cand_widths).max() < MIN_DOOR_WIDTH):
                rows_to_del.append(dp_idx)
        if (len(rows_to_del) < 1):
            return dps


def filter_door_points(dps_on_walls, MIN_DOOR_WIDTH=0.5, MAX_D2D_DIST=0.2):
    final_doors_on_walls = []

    # remove all the singular door points which are too close for both sides
    for dps_idx in range(len(dps_on_walls)):
        dps = dps_on_walls[dps_idx]
        dps_on_walls[dps_idx] = remove_narrow_doors(dps, MIN_DOOR_WIDTH)

    # check head and tail for deletion candidates
    del_candidates = []
    for dps_idx in range(len(dps_on_walls)):
        head_dist, tail_dist = 99, 99
        dps = dps_on_walls[dps_idx]
        if (not dps.any()):
            del_candidates.append([head_dist, tail_dist])
            continue
        pre_dps = dps_on_walls[(dps_idx - 1 + len(dps_on_walls)) % len(dps_on_walls)]
        nxt_dps = dps_on_walls[(dps_idx + 1 + len(dps_on_walls)) % len(dps_on_walls)]
        if (pre_dps.any()):
            head_dist = np.linalg.norm(dps[0, :] - pre_dps[-1, :])
        if (nxt_dps.any()):
            tail_dist = np.linalg.norm(dps[-1, :] - nxt_dps[0, :])
        del_candidates.append([head_dist, tail_dist])

    for dps_idx in range(len(dps_on_walls)):
        dps = dps_on_walls[dps_idx]
        if (not dps.any()):
            continue
        if (dps.shape[0] % 2 == 0):
            res = check_even_dps(dps, del_candidates[dps_idx], MIN_DOOR_WIDTH, MAX_D2D_DIST)
        if (dps.shape[0] % 2 == 1):
            res = check_odd_dps(dps, del_candidates[dps_idx], MIN_DOOR_WIDTH, MAX_D2D_DIST)
        if (res is not None):
            final_doors_on_walls.append(np.array(res))
    return final_doors_on_walls


def get_dist_to_center(x, y, floorW=1024, floorH=512):
    return np.sqrt((x - floorW / 2)**2 + (y - floorH / 2)**2)


def check_inside_range(u0, u_min, u_max):
    if (u_min < u_max):
        if ((u0 > u_min) and (u0 < u_max)):
            return True
        else:
            return False
    if ((u0 > u_min) or (u0 < u_max)):
        return True
    return False


def fuv2img(fuv, coorW=1024, floorW=1024, floorH=512):
    '''
    Project 1d signal in uv space to 2d floor plane image
    '''
    floor_plane_x, floor_plane_y = np.meshgrid(range(floorW), range(floorH))
    floor_plane_x, floor_plane_y = -(floor_plane_y -
                                     floorH / 2), floor_plane_x - floorW / 2
    floor_plane_coridx = (np.arctan2(floor_plane_y, floor_plane_x) /
                          (2 * PI) + 0.5) * coorW - 0.5
    floor_plane = map_coordinates(fuv,
                                  floor_plane_coridx.reshape(1, -1),
                                  order=1,
                                  mode='wrap')
    floor_plane = floor_plane.reshape(floorH, floorW)
    return floor_plane


def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * PI


def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * PI


def np_coor2xy(coor, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    coor: N x 2, index of array in (col, row) format
    '''
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)
    v = np_coory2v(coor[:, 1], coorH)
    c = z / np.tan(v)
    x = c * np.sin(u) + floorW / 2 - 0.5
    y = -c * np.cos(u) + floorH / 2 - 0.5
    return np.hstack([x[:, None], y[:, None]])


def np_x_y_solve_u(x, y, floorW=1024, floorH=512):
    return np.arctan2((y - floorH / 2 + 0.5), (x - floorW / 2 + 0.5))


def np_x_u_solve_y(x, u, floorW=1024, floorH=512):
    c = (x - floorW / 2 + 0.5) / np.sin(u)
    return -c * np.cos(u) + floorH / 2 - 0.5


def np_y_u_solve_x(y, u, floorW=1024, floorH=512):
    c = -(y - floorH / 2 + 0.5) / np.cos(u)
    return c * np.sin(u) + floorW / 2 - 0.5


def np_xy2coor(xy, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    xy: N x 2
    '''
    x = xy[:, 0] - floorW / 2 + 0.5
    y = xy[:, 1] - floorH / 2 + 0.5

    u = np.arctan2(x, -y)
    v = np.arctan(z / np.sqrt(x**2 + y**2))

    coorx = (u / (2 * PI) + 0.5) * coorW - 0.5
    coory = (-v / PI + 0.5) * coorH - 0.5

    return np.hstack([coorx[:, None], coory[:, None]])


def mean_percentile(vec, p1=25, p2=75):
    vmin = np.percentile(vec, p1)
    vmax = np.percentile(vec, p2)
    return vec[(vmin <= vec) & (vec <= vmax)].mean()


def vote(vec, tol):
    vec = np.sort(vec)
    n = np.arange(len(vec))[::-1]
    n = n[:, None] - n[None, :] + 1.0
    l = squareform(pdist(vec[:, None], 'minkowski', p=1) + 1e-9)

    invalid = (n < len(vec) * 0.4) | (l > tol)
    if (~invalid).sum() == 0 or len(vec) < tol:
        best_fit = np.median(vec)
        p_score = 0
    else:
        l[invalid] = 1e5
        n[invalid] = -1
        score = n
        max_idx = score.argmax()
        max_row = max_idx // len(vec)
        max_col = max_idx % len(vec)
        if max_col <= max_row:
            print('vec', len(vec), 'max idx', max_idx)
        best_fit = vec[max_row:max_col + 1].mean()
        p_score = (max_col - max_row + 1) / len(vec)

    l1_score = np.abs(vec - best_fit).mean()

    return best_fit, p_score, l1_score


def get_z1(coory0, coory1, z0=50, coorH=512):
    v0 = np_coory2v(coory0, coorH)
    v1 = np_coory2v(coory1, coorH)
    c0 = z0 / np.tan(v0)
    z1 = c0 * np.tan(v1)
    return z1


def np_refine_by_fix_z(coory0, coory1, z0=50, coorH=512):
    '''
    Refine coory1 by coory0
    coory0 are assumed on given plane z
    '''
    v0 = np_coory2v(coory0, coorH)
    v1 = np_coory2v(coory1, coorH)

    c0 = z0 / np.tan(v0)
    z1 = c0 * np.tan(v1)
    z1_mean = mean_percentile(z1)
    v1_refine = np.arctan2(z1_mean, c0)
    coory1_refine = (-v1_refine / PI + 0.5) * coorH - 0.5

    return coory1_refine, z1_mean


def infer_coory(coory0, h, z0=50, coorH=512):
    v0 = np_coory2v(coory0, coorH)
    c0 = z0 / np.tan(v0)
    z1 = z0 + h
    v1 = np.arctan2(z1, c0)
    return (-v1 / PI + 0.5) * coorH - 0.5


def get_gpid(coorx, coorW):
    gpid = np.zeros(coorW)
    gpid[np.round(coorx).astype(int)] = 1
    gpid = np.cumsum(gpid).astype(int)
    gpid[gpid == gpid[-1]] = 0
    return gpid


def get_gpid_idx(gpid, j):
    idx = np.where(gpid == j)[0]
    if idx[0] == 0 and idx[-1] != len(idx) - 1:
        _shift = -np.where(idx != np.arange(len(idx)))[0][0]
        idx = np.roll(idx, _shift)
    return idx


def gpid_two_split(xy, tpid_a, tpid_b):
    m = np.arange(len(xy)) + 1
    cum_a = np.cumsum(xy[:, tpid_a])
    cum_b = np.cumsum(xy[::-1, tpid_b])
    l1_a = cum_a / m - cum_a / (m * m)
    l1_b = cum_b / m - cum_b / (m * m)
    l1_b = l1_b[::-1]

    score = l1_a[:-1] + l1_b[1:]
    best_split = score.argmax() + 1

    va = xy[:best_split, tpid_a].mean()
    vb = xy[best_split:, tpid_b].mean()

    return va, vb


def _get_rot_rad(px, py):
    if px < 0:
        px, py = -px, -py
    rad = np.arctan2(py, px) * 180 / np.pi
    if rad > 45:
        return 90 - rad
    if rad < -45:
        return -90 - rad
    return -rad


def get_rot_rad(init_coorx, coory, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512, tol=5):
    gpid = get_gpid(init_coorx, coorW)
    coor = np.hstack([np.arange(coorW)[:, None], coory[:, None]])
    xy = np_coor2xy(coor, z, coorW, coorH, floorW, floorH)
    xy_cor = []

    rot_rad_suggestions = []
    for j in range(len(init_coorx)):
        pca = PCA(n_components=1)
        pca.fit(xy[gpid == j])
        rot_rad_suggestions.append(_get_rot_rad(*pca.components_[0]))
    rot_rad_suggestions = np.sort(rot_rad_suggestions + [1e9])

    rot_rad = np.mean(rot_rad_suggestions[:-1])
    best_rot_rad_sz = -1
    last_j = 0
    for j in range(1, len(rot_rad_suggestions)):
        if rot_rad_suggestions[j] - rot_rad_suggestions[j - 1] > tol:
            last_j = j
        elif j - last_j > best_rot_rad_sz:
            rot_rad = rot_rad_suggestions[last_j:j + 1].mean()
            best_rot_rad_sz = j - last_j

    dx = int(round(rot_rad * 1024 / 360))
    return dx, rot_rad


def gen_ww_cuboid(xy, gpid, tol):
    xy_cor = []
    assert len(np.unique(gpid)) == 4

    # For each part seperated by wall-wall peak, voting for a wall
    for j in range(4):
        now_x = xy[gpid == j, 0]
        now_y = xy[gpid == j, 1]
        new_x, x_score, x_l1 = vote(now_x, tol)
        new_y, y_score, y_l1 = vote(now_y, tol)
        if (x_score, -x_l1) > (y_score, -y_l1):
            xy_cor.append({'type': 0, 'val': new_x, 'score': x_score})
        else:
            xy_cor.append({'type': 1, 'val': new_y, 'score': y_score})

    # Sanity fallback
    scores = [0, 0]
    for j in range(4):
        if xy_cor[j]['type'] == 0:
            scores[j % 2] += xy_cor[j]['score']
        else:
            scores[j % 2] -= xy_cor[j]['score']
    if scores[0] > scores[1]:
        xy_cor[0]['type'] = 0
        xy_cor[1]['type'] = 1
        xy_cor[2]['type'] = 0
        xy_cor[3]['type'] = 1
    else:
        xy_cor[0]['type'] = 1
        xy_cor[1]['type'] = 0
        xy_cor[2]['type'] = 1
        xy_cor[3]['type'] = 0

    return xy_cor


def gen_left_right_doors(cor_id, y_seg_wd_, y_seg_dw_, r, min_v, cam_height=1.6, W=1024, H=512):

    signal_door_head = y_seg_wd_
    pk_loc_head = get_maxi_loc(signal_door_head, min_v, r)

    signal_door_tail = y_seg_dw_
    pk_loc_tail = get_maxi_loc(signal_door_tail, min_v, r)

    ww_loc = cor_id[::2, 0].astype(int)
    floor_z = -cam_height
    floor_xy = np_coor2xy(cor_id[1::2], floor_z, W, H, floorW=1, floorH=1)

    u = np_coorx2u(cor_id[1::2, 0], W)
    signal_full = y_seg_
    doors_on_wall = []
    doors_score = []

    for idx in range(0, ww_loc.shape[0]):
        start_xy = floor_xy[idx, :]
        end_xy = floor_xy[(idx + 1) % ww_loc.shape[0], :]
        if (np.sum(abs(start_xy - end_xy)) < 0.6):
            continue
        if (idx == ww_loc.shape[0] - 1):
            signal1 = signal_full[np.arange(ww_loc[-1], W)]
            signal2 = signal_full[np.arange(0, ww_loc[0])]
            pk_loc1 = get_maxi_loc(signal1, min_v, r)
            pk_loc1 += ww_loc[-1]
            pk_loc2 = get_maxi_loc(signal2, min_v, r)
            pk_loc = np.concatenate([pk_loc1, pk_loc2])
            #print('pk', pk_loc1, pk_loc2)
        else:
            signal = signal_full[np.arange(ww_loc[idx], ww_loc[idx + 1])]
            pk_loc = get_maxi_loc(signal, min_v, r)
            pk_loc += ww_loc[idx]

        #max_v = max_v[pk_loc]
        #if(idx == ww_loc.shape[0] - 1):
        #    pk_loc = pk_loc % W
        u_doors = np_coorx2u(pk_loc, W)
        door_xys = []
        for u_door in u_doors:
            door_xy = interp_point_by_u(start_xy, end_xy, u[idx], u[(idx + 1) % ww_loc.shape[0]], u_door)
            door_xys.append(door_xy)
        door_xys = np.array(door_xys)
        doors_on_wall.append(door_xys)
    return doors_on_wall


def gen_ww_general(init_coorx, xy, gpid, tol):
    xy_cor = []
    assert len(init_coorx) == len(np.unique(gpid))

    # Candidate for each part seperated by wall-wall boundary
    for j in range(len(init_coorx)):
        now_x = xy[gpid == j, 0]
        now_y = xy[gpid == j, 1]
        if (len(now_x) < 2):
            continue
        new_x, x_score, x_l1 = vote(now_x, tol)
        new_y, y_score, y_l1 = vote(now_y, tol)
        u0 = np_coorx2u(init_coorx[(j - 1 + len(init_coorx)) % len(init_coorx)])
        u1 = np_coorx2u(init_coorx[j])
        if (x_score, -x_l1) > (y_score, -y_l1):
            xy_cor.append({
                'type': 0,
                'val': new_x,
                'score': x_score,
                'action': 'ori',
                'gpid': j,
                'u0': u0,
                'u1': u1,
                'tbd': True
            })
        else:
            xy_cor.append({
                'type': 1,
                'val': new_y,
                'score': y_score,
                'action': 'ori',
                'gpid': j,
                'u0': u0,
                'u1': u1,
                'tbd': True
            })

    # Construct wall from highest score to lowest
    while True:
        # Finding undetermined wall with highest score
        tbd = -1
        for i in range(len(xy_cor)):
            if xy_cor[i]['tbd'] and (tbd == -1 or xy_cor[i]['score'] > xy_cor[tbd]['score']):
                tbd = i
        if tbd == -1:
            break

        # This wall is determined
        xy_cor[tbd]['tbd'] = False
        p_idx = (tbd - 1 + len(xy_cor)) % len(xy_cor)
        n_idx = (tbd + 1) % len(xy_cor)

        num_tbd_neighbor = xy_cor[p_idx]['tbd'] + xy_cor[n_idx]['tbd']

        # Two adjacency walls are not determined yet => not special case
        if num_tbd_neighbor == 2:
            continue

        # Only one of adjacency two walls is determine => add now or later special case
        if num_tbd_neighbor == 1:
            if (not xy_cor[p_idx]['tbd'] and xy_cor[p_idx]['type'] == xy_cor[tbd]['type']) or\
                    (not xy_cor[n_idx]['tbd'] and xy_cor[n_idx]['type'] == xy_cor[tbd]['type']):
                # Current wall is different from one determined adjacency wall
                if xy_cor[tbd]['score'] >= -1:
                    # Later special case, add current to tbd
                    xy_cor[tbd]['tbd'] = True
                    xy_cor[tbd]['score'] -= 100
                else:
                    # Fallback: forced change the current wall or infinite loop
                    if not xy_cor[p_idx]['tbd']:
                        insert_at = tbd
                        if xy_cor[p_idx]['type'] == 0:
                            new_val = np_x_u_solve_y(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                            new_type = 1
                        else:
                            new_val = np_y_u_solve_x(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                            new_type = 0
                    else:
                        insert_at = n_idx
                        if xy_cor[n_idx]['type'] == 0:
                            new_val = np_x_u_solve_y(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
                            new_type = 1
                        else:
                            new_val = np_y_u_solve_x(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
                            new_type = 0
                    new_add = {
                        'type': new_type,
                        'val': new_val,
                        'score': 0,
                        'action': 'forced infer',
                        'gpid': -1,
                        'u0': -1,
                        'u1': -1,
                        'tbd': False
                    }
                    xy_cor.insert(insert_at, new_add)
            continue

        # Below checking special case
        if xy_cor[p_idx]['type'] == xy_cor[n_idx]['type']:
            # Two adjacency walls are same type, current wall should be differen type
            if xy_cor[tbd]['type'] == xy_cor[p_idx]['type']:
                # Fallback: three walls with same type => forced change the middle wall
                xy_cor[tbd]['type'] = (xy_cor[tbd]['type'] + 1) % 2
                xy_cor[tbd]['action'] = 'forced change'
                xy_cor[tbd]['val'] = xy[gpid == xy_cor[tbd]['gpid'], xy_cor[tbd]['type']].mean()
        else:
            # Two adjacency walls are different type => add one
            tp0 = xy_cor[n_idx]['type']
            tp1 = xy_cor[p_idx]['type']
            if xy_cor[p_idx]['type'] == 0:
                val0 = np_x_u_solve_y(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                val1 = np_y_u_solve_x(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
            else:
                val0 = np_y_u_solve_x(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                val1 = np_x_u_solve_y(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
            new_add = [
                {
                    'type': tp0,
                    'val': val0,
                    'score': 0,
                    'action': 'forced infer',
                    'gpid': -1,
                    'u0': -1,
                    'u1': -1,
                    'tbd': False
                },
                {
                    'type': tp1,
                    'val': val1,
                    'score': 0,
                    'action': 'forced infer',
                    'gpid': -1,
                    'u0': -1,
                    'u1': -1,
                    'tbd': False
                },
            ]
            xy_cor = xy_cor[:tbd] + new_add + xy_cor[tbd + 1:]

    return xy_cor


def gen_ww_general_with_order_constraints(init_coorx, xy, gpid, tol):
    xy_cor = []
    if (len(init_coorx) != len(np.unique(gpid))):
        print('init_coorx', init_coorx, 'gpid len', len(np.unique(gpid)))

    # Candidate for each part seperated by wall-wall boundary
    for j in range(len(init_coorx)):
        now_x = xy[gpid == j, 0]
        now_y = xy[gpid == j, 1]
        if (len(now_x)) < 2:
            continue
        new_x, x_score, x_l1 = vote(now_x, tol)
        new_y, y_score, y_l1 = vote(now_y, tol)
        u0 = np_coorx2u(init_coorx[(j - 1 + len(init_coorx)) % len(init_coorx)])
        u1 = np_coorx2u(init_coorx[j])
        if (x_score, -x_l1) > (y_score, -y_l1):
            xy_cor.append({
                'type': 0,
                'val': new_x,
                'score': x_score,
                'action': 'ori',
                'gpid': j,
                'u0': u0,
                'u1': u1,
                'tbd': True
            })
        else:
            xy_cor.append({
                'type': 1,
                'val': new_y,
                'score': y_score,
                'action': 'ori',
                'gpid': j,
                'u0': u0,
                'u1': u1,
                'tbd': True
            })

    # Construct wall from highest score to lowest
    while True:
        # Finding undetermined wall with highest score
        tbd = -1
        for i in range(len(xy_cor)):
            if xy_cor[i]['tbd'] and (tbd == -1 or xy_cor[i]['score'] > xy_cor[tbd]['score']):
                tbd = i
        if tbd == -1:
            break

        # This wall is determined
        xy_cor[tbd]['tbd'] = False
        p_idx = (tbd - 1 + len(xy_cor)) % len(xy_cor)
        n_idx = (tbd + 1) % len(xy_cor)

        num_tbd_neighbor = xy_cor[p_idx]['tbd'] + xy_cor[n_idx]['tbd']

        # Two adjacency walls are not determined yet => not special case
        if num_tbd_neighbor == 2:
            continue

        # Only one of adjacency two walls is determine => add now or later special case
        if num_tbd_neighbor == 1:
            #print('test1')
            if (not xy_cor[p_idx]['tbd'] and xy_cor[p_idx]['type'] == xy_cor[tbd]['type']) or\
                    (not xy_cor[n_idx]['tbd'] and xy_cor[n_idx]['type'] == xy_cor[tbd]['type']):
                # Current wall is different from one determined adjacency wall
                if xy_cor[tbd]['score'] >= -1:
                    # Later special case, add current to tbd
                    xy_cor[tbd]['tbd'] = True
                    xy_cor[tbd]['score'] -= 100
                else:
                    # Fallback: forced change the current wall or infinite loop
                    if not xy_cor[p_idx]['tbd']:
                        insert_at = tbd
                        if xy_cor[p_idx]['type'] == 0:
                            new_val = np_x_u_solve_y(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                            new_type = 1
                        else:
                            new_val = np_y_u_solve_x(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                            new_type = 0
                    else:
                        insert_at = n_idx
                        if xy_cor[n_idx]['type'] == 0:
                            new_val = np_x_u_solve_y(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
                            new_type = 1
                        else:
                            new_val = np_y_u_solve_x(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
                            new_type = 0
                    new_add = {
                        'type': new_type,
                        'val': new_val,
                        'score': 0,
                        'action': 'forced infer',
                        'gpid': -1,
                        'u0': -1,
                        'u1': -1,
                        'tbd': False
                    }
                    xy_cor.insert(insert_at, new_add)
            continue

        # Below checking special case
        if xy_cor[p_idx]['type'] == xy_cor[n_idx]['type']:
            # Two adjacency walls are same type, current wall should be differen type
            if xy_cor[tbd]['type'] == xy_cor[p_idx]['type']:
                # Fallback: three walls with same type => forced change the middle wall
                xy_cor[tbd]['type'] = (xy_cor[tbd]['type'] + 1) % 2
                xy_cor[tbd]['action'] = 'forced change'
                xy_cor[tbd]['val'] = xy[gpid == xy_cor[tbd]['gpid'], xy_cor[tbd]['type']].mean()
        else:
            # Two adjacency walls are different type => add one
            #print(xy_cor[p_idx], xy_cor[tbd], xy_cor[n_idx])
            tp0 = xy_cor[n_idx]['type']
            tp1 = xy_cor[p_idx]['type']
            if xy_cor[p_idx]['type'] == 0:
                val0 = np_x_u_solve_y(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                val1 = np_y_u_solve_x(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
                insert_corner_u = np_x_y_solve_u(val1, val0)
                insert_dist = get_dist_to_center(val1, val0)
                pre_dist = get_dist_to_center(xy_cor[p_idx]['val'], val0)
                nxt_dist = get_dist_to_center(val1, xy_cor[n_idx]['val'])
            else:
                val0 = np_y_u_solve_x(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                val1 = np_x_u_solve_y(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
                insert_corner_u = np_x_y_solve_u(val0, val1)
                insert_dist = get_dist_to_center(val0, val1)
                pre_dist = get_dist_to_center(val0, xy_cor[p_idx]['val'])
                nxt_dist = get_dist_to_center(xy_cor[n_idx]['val'], val1)

            if (check_inside_range(insert_corner_u, xy_cor[tbd]['u0'], xy_cor[tbd]['u1'])):
                # no occlusion happened
                new_add = [
                    {
                        'type': tp0,
                        'val': val0,
                        'score': 0,
                        'action': 'forced infer',
                        'gpid': -1,
                        'u0': -1,
                        'u1': -1,
                        'tbd': False
                    },
                    {
                        'type': tp1,
                        'val': val1,
                        'score': 0,
                        'action': 'forced infer',
                        'gpid': -1,
                        'u0': -1,
                        'u1': -1,
                        'tbd': False
                    },
                ]
                xy_cor = xy_cor[:tbd] + new_add + xy_cor[tbd + 1:]
            else:
                dist_to_compare = pre_dist if (insert_corner_u < xy_cor[tbd]['u0']) else nxt_dist
                if (insert_dist > dist_to_compare):
                    new_add = [
                        {
                            'type': tp0,
                            'val': val0,
                            'score': 0,
                            'action': 'forced infer',
                            'gpid': -1,
                            'u0': -1,
                            'u1': -1,
                            'tbd': False
                        },
                        {
                            'type': tp1,
                            'val': val1,
                            'score': 0,
                            'action': 'forced infer',
                            'gpid': -1,
                            'u0': -1,
                            'u1': -1,
                            'tbd': False
                        },
                    ]
                    xy_cor = xy_cor[:tbd] + new_add + xy_cor[tbd + 1:]
                else:
                    xy_cor = xy_cor[:tbd] + xy_cor[tbd + 1:]
    return xy_cor


def gen_ww(init_coorx,
           coory,
           z=50,
           coorW=1024,
           coorH=512,
           floorW=1024,
           floorH=512,
           tol=3,
           force_cuboid=True):
    init_coorx = np.delete(init_coorx, np.where(abs(init_coorx[1:]-init_coorx[:-1]) <1)[0]+1, 0)
    gpid = get_gpid(init_coorx, coorW)
    coor = np.hstack([np.arange(coorW)[:, None], coory[:, None]])
    xy = np_coor2xy(coor, z, coorW, coorH, floorW, floorH)

    # Generate wall-wall
    if force_cuboid:
        xy_cor = gen_ww_cuboid(xy, gpid, tol)
    else:
        xy_cor = gen_ww_general_with_order_constraints(init_coorx, xy, gpid, tol)
        #xy_cor = gen_ww_general(init_coorx, xy, gpid, tol)

    # Ceiling view to normal view
    cor = []
    for j in range(len(xy_cor)):
        next_j = (j + 1) % len(xy_cor)
        if xy_cor[j]['type'] == 1:
            cor.append((xy_cor[next_j]['val'], xy_cor[j]['val']))
        else:
            cor.append((xy_cor[j]['val'], xy_cor[next_j]['val']))
    cor = np_xy2coor(np.array(cor), z, coorW, coorH, floorW, floorH)
    cor = np.roll(cor, -2 * cor[::2, 0].argmin(), axis=0)

    return cor, xy_cor


def create_door_xys(floor_xy, xs_seg_wd_, xs_seg_dw_, ww_loc_padded, floor_z, u, W):
    # Fix near wall-wall corner door boundary bug
    xs_seg_wd_[np.where(xs_seg_wd_.reshape(-1, 1) in ww_loc_padded.reshape(1, -1))] += 1
    xs_seg_dw_[np.where(xs_seg_dw_.reshape(-1, 1) in ww_loc_padded.reshape(1, -1))] -= 1

    wd_bounds_left = np.sign(xs_seg_wd_.reshape(-1, 1) - ww_loc_padded.reshape(1, -1))
    wd_bounds_right = np.sign(xs_seg_wd_.reshape(-1, 1) - np.roll(ww_loc_padded, -1, axis=0).reshape(1, -1))
    indicator_wd = wd_bounds_right + wd_bounds_left
    indicator_wd[:, 0] = indicator_wd[:, 0] * indicator_wd[:, -2]
    indicator_wd = indicator_wd[:, :-2]
    indicator_wd = np.roll(indicator_wd, -1, axis=1)

    dw_bounds_left = np.sign(xs_seg_dw_.reshape(-1, 1) - ww_loc_padded.reshape(1, -1))
    dw_bounds_right = np.sign(xs_seg_dw_.reshape(-1, 1) - np.roll(ww_loc_padded, -1, axis=0).reshape(1, -1))
    indicator_dw = dw_bounds_right + dw_bounds_left
    indicator_dw[:, 0] = indicator_dw[:, 0] * indicator_dw[:, -2]
    indicator_dw = indicator_dw[:, :-2]
    indicator_dw = np.roll(indicator_dw, -1, axis=1)

    u_doors_lb = np_coorx2u(xs_seg_wd_, W)
    u_doors_rb = np_coorx2u(xs_seg_dw_, W)
    door_xys_lb = []
    door_xys_rb = []
    for idx in range(0, floor_xy.shape[0]):
        start_xy = floor_xy[idx, :]
        end_xy = floor_xy[(idx + 1) % floor_xy.shape[0], :]
        ud_lb_selected = u_doors_lb[np.where(
            indicator_wd[:, idx].reshape(-1) == 0)]
        ud_rb_selected = u_doors_rb[np.where(
            indicator_dw[:, idx].reshape(-1) == 0)]

        for u_door in ud_lb_selected:
            door_xy = interp_point_by_u(start_xy, end_xy, u[idx],
                                        u[(idx + 1) % floor_xy.shape[0]],
                                        u_door)
            door_xys_lb.append(np.append(door_xy, [u_door, idx]))
        for u_door in ud_rb_selected:
            door_xy = interp_point_by_u(start_xy, end_xy, u[idx],
                                        u[(idx + 1) % floor_xy.shape[0]],
                                        u_door)
            door_xys_rb.append(np.append(door_xy, [u_door, idx]))
    return door_xys_lb, door_xys_rb


def gen_doors_from_ordered_segs(cor_id,
                                xs_seg_wd_,
                                xs_seg_dw_,
                                lb_scores,
                                rb_scores,
                                W=1024,
                                H=512,
                                cam_height=1.6):
    # image mirror effect, so we flip the y-axis, then left bound and right bound are swapped
    # ww_loc = cor_id[::2, 0].astype(int)
    final_door_xys = []
    ww_loc_padded = np.append(np.append(0, cor_id[::2, 0].astype(int)), W)
    floor_z = -cam_height
    floor_xy = np_coor2xy(cor_id[1::2], floor_z, W, H, floorW=1, floorH=1)
    u = np_coorx2u(cor_id[1::2, 0], W)
    if((xs_seg_dw_.shape[0] == 0) or (xs_seg_wd_.shape[0] == 0)):
        return final_door_xys
    roll_max_len = xs_seg_dw_.shape[0]
    while ((xs_seg_wd_[0] > xs_seg_dw_[0]) and roll_max_len>0):
        xs_seg_dw_ = np.roll(xs_seg_dw_, -1, axis=0)
        rb_scores = np.roll(rb_scores, -1, axis=0)
        roll_max_len -= 1
    door_xys_lb, door_xys_rb = create_door_xys(floor_xy, xs_seg_wd_,
                                               xs_seg_dw_, ww_loc_padded,
                                               floor_z, u, W)

    lb_idx, rb_idx = 0, 0
    ori_door_point_num = len(door_xys_rb) + len(door_xys_rb)
    fail_count = ori_door_point_num * 2 + 1
    repeat_check_l = 0
    repeat_check_r = 0
    same_wall_threshold = 0.05

    while (len(door_xys_lb) > 0 and len(door_xys_rb) > 0 and fail_count > 0):
        lb_idx = lb_idx % len(door_xys_lb)
        rb_idx = rb_idx % len(door_xys_rb)
        fail_count -= 1
        if (fail_count == ori_door_point_num):
            same_wall_threshold = same_wall_threshold * 1.5
        if (fail_count == ori_door_point_num // 2):
            same_wall_threshold = same_wall_threshold * 1.5

        door_lb = door_xys_lb[lb_idx]
        door_rb = door_xys_rb[rb_idx]
        door_vec = -door_rb[:2] + door_lb[:2]
        same_wall_check = min(abs(door_vec)) < same_wall_threshold
        if (not same_wall_check):
            # not the same wall, so push left boundary
            #if ((lb_scores[lb_idx] > rb_scores[rb_idx])
            if (repeat_check_l < len(door_xys_lb)):
                rb_idx += 1
                repeat_check_l += 1
                repeat_check_r = 0
            else:
                lb_idx += 1
                repeat_check_r += 1
                repeat_check_l = 0
            continue
        axis = np.argmin(abs(door_vec))
        if (axis == 0):
            bounds_order_check = (door_lb[0] > 0
                                  and door_vec[1] < 0) or (door_lb[0] < 0
                                                           and door_vec[1] > 0)
        else:
            bounds_order_check = (door_lb[1] > 0 and door_vec[0] > 0) or (door_lb[1] < 0 and door_vec[0] < 0)
        if (not bounds_order_check):
            rb_idx += 1
            repeat_check_l += 1
            repeat_check_r = 0
            continue
        min_width_check = max(abs(door_vec)) > 0.2
        if (not min_width_check):
            rb_idx += 1
            repeat_check_l += 1
            repeat_check_r = 0
            continue
        max_width_check = max(abs(door_vec)) < 3.5
        if (not max_width_check):
            lb_idx += 1
            repeat_check_r += 1
            repeat_check_l = 0
            continue
        final_door_xys.append(door_xys_lb[lb_idx][:2])
        door_xys_lb.pop(lb_idx)
        # lb_scores = np.delete(lb_scores, lb_idx)

        final_door_xys.append(door_xys_rb[rb_idx][:2])
        door_xys_rb.pop(rb_idx)
        # rb_scores = np.delete(rb_scores, rb_idx)
    return final_door_xys