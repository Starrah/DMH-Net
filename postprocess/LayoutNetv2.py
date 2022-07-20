import numpy as np
import scipy.signal
import torch
from scipy.ndimage.filters import maximum_filter
from torch import optim
import postprocess.LayoutNet_post_proc2 as post_proc
from scipy.ndimage import convolve, map_coordinates
from shapely.geometry import Polygon


def LayoutNetv2PostProcessMain(cor_img: np.ndarray, edg_img: np.ndarray) -> np.ndarray:
    """
    :param cor_img ndarray<512,1024>
    :param edg_img ndarray<512,1024,3>
    """
    # general layout, tp view
    cor_ = cor_img.sum(0)
    cor_ = (cor_ - np.amin(cor_)) / np.ptp(cor_)
    min_v = 0.25  # 0.05
    xs_ = find_N_peaks(cor_, r=26, min_v=min_v, N=None)[0]
    # spetial case for too less corner
    if xs_.shape[0] < 4:
        xs_ = find_N_peaks(cor_, r=26, min_v=0.05, N=None)[0]
        if xs_.shape[0] < 4:
            xs_ = find_N_peaks(cor_, r=26, min_v=0, N=None)[0]
    # get ceil and floor line
    ceil_img = edg_img[:, :, 1]
    floor_img = edg_img[:, :, 2]
    ceil_idx = np.argmax(ceil_img, axis=0)
    floor_idx = np.argmax(floor_img, axis=0)
    # Init floor/ceil plane
    z0 = 50
    force_cuboid = False
    _, z1 = post_proc.np_refine_by_fix_z(ceil_idx, floor_idx, z0)
    # Generate general  wall-wall
    cor, xy_cor = post_proc.gen_ww(xs_, ceil_idx, z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=force_cuboid)

    if not force_cuboid:
        # Check valid (for fear self-intersection)
        xy2d = np.zeros((len(xy_cor), 2), np.float32)
        for i in range(len(xy_cor)):
            xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
            xy2d[i, xy_cor[i - 1]['type']] = xy_cor[i - 1]['val']
        if not Polygon(xy2d).is_valid:
            # actually it's not force cuboid, just assume all corners are visible, go back to original LayoutNet initialization
            # print(
            #    'Fail to generate valid general layout!! '
            #    'Generate cuboid as fallback.',
            #    file=sys.stderr)
            cor_id = get_ini_cor(cor_img, 21, 3)
            force_cuboid = True

    if not force_cuboid:
        # Expand with btn coory
        cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])
        # Collect corner position in equirectangular
        cor_id = np.zeros((len(cor) * 2, 2), np.float32)
        for j in range(len(cor)):
            cor_id[j * 2] = cor[j, 0], cor[j, 1]
            cor_id[j * 2 + 1] = cor[j, 0], cor[j, 2]

    # refinement
    cor_id = optimize_cor_id(cor_id, edg_img, cor_img, num_iters=100, verbose=False)

    return cor_id


def find_N_peaks(signal, r=29, min_v=0.05, N=None):
    max_v = maximum_filter(signal, size=r, mode='wrap')
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    # check for odd case, remove one
    if (pk_loc.shape[0]%2)!=0:
        pk_id = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[pk_id[:-1]]
        pk_loc = np.sort(pk_loc)
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]
    return pk_loc, signal[pk_loc]


def get_ini_cor(cor_img, d1=21, d2=3):
    cor = convolve(cor_img, np.ones((d1, d1)), mode='constant', cval=0.0)
    cor_id = []
    cor_ = cor_img.sum(0)
    cor_ = (cor_ - np.amin(cor_)) / np.ptp(cor_)

    min_v = 0.25  # 0.05
    xs_ = find_N_peaks(cor_, r=26, min_v=min_v, N=None)[0]

    # spetial case for too less corner
    if xs_.shape[0] < 4:
        xs_ = find_N_peaks(cor_, r=26, min_v=0.05, N=None)[0]
        if xs_.shape[0] < 4:
            xs_ = find_N_peaks(cor_, r=26, min_v=0, N=None)[0]

    X_loc = xs_
    for x in X_loc:
        x_ = int(np.round(x))

        V_signal = cor[:, max(0, x_ - d2):x_ + d2 + 1].sum(1)
        y1, y2 = find_N_peaks_conv(V_signal, prominence=None,
                                   distance=20, N=2)[0]
        cor_id.append((x, y1))
        cor_id.append((x, y2))

    cor_id = np.array(cor_id, np.float64)

    return cor_id


def find_N_peaks_conv(signal, prominence, distance, N=4):
    locs, _ = scipy.signal.find_peaks(signal,
                                      prominence=prominence,
                                      distance=distance)
    pks = signal[locs]
    pk_id = np.argsort(-pks)
    pk_loc = locs[pk_id[:min(N, len(pks))]]
    pk_loc = np.sort(pk_loc)
    return pk_loc, signal[pk_loc]


def optimize_cor_id(cor_id, scoreedg, scorecor, num_iters=100, verbose=False):
    assert scoreedg.shape == (512, 1024, 3)
    assert scorecor.shape == (512, 1024)

    Z = -1
    ceil_cor_id = cor_id[0::2]
    floor_cor_id = cor_id[1::2]

    ceil_cor_id, ceil_cor_id_xy = constraint_cor_id_same_z(ceil_cor_id, scorecor, Z)
    # ceil_cor_id_xyz = np.hstack([ceil_cor_id_xy, np.zeros(4).reshape(-1, 1) + Z])
    ceil_cor_id_xyz = np.hstack([ceil_cor_id_xy, np.zeros(ceil_cor_id.shape[0]).reshape(-1, 1) + Z])

    # TODO: revise here to general layout
    # pc = (ceil_cor_id_xy[0] + ceil_cor_id_xy[2]) / 2
    # print(ceil_cor_id_xy)
    if abs(ceil_cor_id_xy[0, 0] - ceil_cor_id_xy[1, 0]) > abs(ceil_cor_id_xy[0, 1] - ceil_cor_id_xy[1, 1]):
        ceil_cor_id_xy = np.concatenate((ceil_cor_id_xy[1:, :], ceil_cor_id_xy[:1, :]), axis=0)
    # print(cor_id)
    # print(ceil_cor_id_xy)
    pc = np.mean(ceil_cor_id_xy, axis=0)
    pc_vec = ceil_cor_id_xy[0] - pc
    pc_theta = vecang(pc_vec, ceil_cor_id_xy[1] - pc)
    pc_height = fit_avg_z(floor_cor_id, ceil_cor_id_xy, scorecor)

    if ceil_cor_id_xy.shape[0] > 4:
        pc_theta = np.array([ceil_cor_id_xy[1, 1]])
        for c_num in range(2, ceil_cor_id_xy.shape[0] - 1):
            if (c_num % 2) == 0:
                pc_theta = np.append(pc_theta, ceil_cor_id_xy[c_num, 0])
            else:
                pc_theta = np.append(pc_theta, ceil_cor_id_xy[c_num, 1])

    with torch.enable_grad():
        scoreedg = torch.FloatTensor(scoreedg)
        scorecor = torch.FloatTensor(scorecor)
        pc = torch.FloatTensor(pc)
        pc_vec = torch.FloatTensor(pc_vec)
        pc_theta = torch.FloatTensor([pc_theta])
        pc_height = torch.FloatTensor([pc_height])
        pc.requires_grad = True
        pc_vec.requires_grad = True
        pc_theta.requires_grad = True
        pc_height.requires_grad = True

        # print(pc_theta)
        # time.sleep(2)
        # return cor_id
        optimizer = optim.SGD([
            pc, pc_vec, pc_theta, pc_height
        ], lr=1e-3, momentum=0.9)

        best = {'score': 1e9}

        for i_step in range(num_iters):
            i = i_step if verbose else None
            optimizer.zero_grad()
            score = project2sphere_score(pc, pc_vec, pc_theta, pc_height, scoreedg, scorecor, i)
            if score.item() < best['score']:
                best['score'] = score.item()
                best['pc'] = pc.clone()
                best['pc_vec'] = pc_vec.clone()
                best['pc_theta'] = pc_theta.clone()
                best['pc_height'] = pc_height.clone()
            score.backward()
            optimizer.step()

    pc = best['pc']
    pc_vec = best['pc_vec']
    pc_theta = best['pc_theta']
    pc_height = best['pc_height']
    opt_cor_id = pc2cor_id(pc, pc_vec, pc_theta, pc_height).detach().numpy()
    split_num = int(opt_cor_id.shape[0] // 2)
    opt_cor_id = np.stack([opt_cor_id[:split_num], opt_cor_id[split_num:]], axis=1).reshape(split_num * 2, 2)

    # print(opt_cor_id)
    # print(cor_id)
    # time.sleep(500)
    return opt_cor_id


def constraint_cor_id_same_z(cor_id, cor_img, z=-1):
    # Convert to uv space
    cor_id_u = ((cor_id[:, 0] + 0.5) / cor_img.shape[1] - 0.5) * 2 * np.pi
    cor_id_v = ((cor_id[:, 1] + 0.5) / cor_img.shape[0] - 0.5) * np.pi

    # Convert to xyz space (z=-1)
    cor_id_c = z / np.tan(cor_id_v)
    cor_id_xy = np.stack([
        cor_id_c * np.cos(cor_id_u),
        cor_id_c * np.sin(cor_id_u),
    ], axis=0).T

#    # Fix 2 diagonal corner, move the others
#    cor_id_score = map_coordinates(cor_img, [cor_id[:, 1], cor_id[:, 0]])
#    if cor_id_score[0::2].sum() > cor_id_score[1::2].sum():
#        idx0, idx1 = 0, 1
#    else:
#        idx0, idx1 = 1, 0
#    pc = cor_id_xy[idx0::2].mean(0, keepdims=True)
#    radius2 = np.sqrt(((cor_id_xy[idx0::2] - pc) ** 2).sum(1)).mean()
#    d = cor_id_xy[idx1::2] - pc
#    d1 = d[0]
#    d2 = d[1]
#    theta1 = (np.arctan2(d1[1], d1[0]) + 2 * np.pi) % (2 * np.pi)
#    theta2 = (np.arctan2(d2[1], d2[0]) + 2 * np.pi) % (2 * np.pi)
#    theta2 = theta2 - np.pi
#    theta2 = (theta2 + 2 * np.pi) % (2 * np.pi)
#    theta = (theta1 + theta2) / 2
#    d[0] = (radius2 * np.cos(theta), radius2 * np.sin(theta))
#    theta = theta - np.pi
#    d[1] = (radius2 * np.cos(theta), radius2 * np.sin(theta))

#    cor_id_xy[idx1::2] = pc + d

    # Convert refined xyz back to uv space
    cor_id_uv = np.stack([
        np.arctan2(cor_id_xy[:, 1], cor_id_xy[:, 0]),
        np.arctan2(z, np.sqrt((cor_id_xy ** 2).sum(1))),
    ], axis=0).T

    # Convert to image index
    col = (cor_id_uv[:, 0] / (2 * np.pi) + 0.5) * cor_img.shape[1] - 0.5
    row = (cor_id_uv[:, 1] / np.pi + 0.5) * cor_img.shape[0] - 0.5
    return np.stack([col, row], axis=0).T, cor_id_xy


def fit_avg_z(cor_id, cor_id_xy, cor_img):
    score = map_coordinates(cor_img, [cor_id[:, 1], cor_id[:, 0]])
    c = np.sqrt((cor_id_xy ** 2).sum(1))
    cor_id_v = ((cor_id[:, 1] + 0.5) / cor_img.shape[0] - 0.5) * np.pi
    z = c * np.tan(cor_id_v)
    fit_z = (z * score).sum() / score.sum()
    return fit_z


def map_coordinates_Pytorch(input, coordinates):
    ''' PyTorch version of scipy.ndimage.interpolation.map_coordinates
    input: (H, W)
    coordinates: (2, ...)
    '''
    h = input.shape[0]
    w = input.shape[1]

    def _coordinates_pad_wrap(h, w, coordinates):
        coordinates[0] = coordinates[0] % h
        coordinates[1] = coordinates[1] % w
        return coordinates

    co_floor = torch.floor(coordinates).long()
    co_ceil = torch.ceil(coordinates).long()
    d1 = (coordinates[1] - co_floor[1].float())
    d2 = (coordinates[0] - co_floor[0].float())
    co_floor = _coordinates_pad_wrap(h, w, co_floor)
    co_ceil = _coordinates_pad_wrap(h, w, co_ceil)
    f00 = input[co_floor[0], co_floor[1]]
    f10 = input[co_floor[0], co_ceil[1]]
    f01 = input[co_ceil[0], co_floor[1]]
    f11 = input[co_ceil[0], co_ceil[1]]
    fx1 = f00 + d1 * (f10 - f00)
    fx2 = f01 + d1 * (f11 - f01)
    return fx1 + d2 * (fx2 - fx1)


def project2sphere_score(pc, pc_vec, pc_theta, pc_height, scoreedg, scorecor, i_step=None):

    # Sample corner loss
    corid = pc2cor_id(pc, pc_vec, pc_theta, pc_height)
    corid_coordinates = torch.stack([corid[:, 1], corid[:, 0]])
    loss_cor = -map_coordinates_Pytorch(scorecor, corid_coordinates).mean()

    # Sample boundary loss
    if pc_theta.numel()==1:
        p1 = pc + pc_vec
        p2 = pc + rotatevec(pc_vec, pc_theta)
        p3 = pc - pc_vec
        p4 = pc + rotatevec(pc_vec, pc_theta - np.pi)

        segs = [
            pts_linspace(p1, p2),
            pts_linspace(p2, p3),
            pts_linspace(p3, p4),
            pts_linspace(p4, p1),
        ]
    else:
        ps = pc + pc_vec
        ps = ps.view(-1,2)
        for c_num in range(pc_theta.shape[1]):
            ps = torch.cat((ps, ps[c_num:,:]),0)
            if (c_num % 2) == 0:
                ps[-1,1] = pc_theta[0,c_num]
            else:
                ps[-1,0] = pc_theta[0,c_num]
        ps = torch.cat((ps, ps[-1:,:]),0)
        ps[-1,1] = ps[0,1]
        segs = []
        for c_num in range(ps.shape[0]-1):
            segs.append(pts_linspace(ps[c_num,:], ps[c_num+1,:]))
        segs.append(pts_linspace(ps[-1,:], ps[0,:]))

    # ceil-wall
    loss_ceilwall = 0
    for seg in segs:
        ceil_uv = xyz2uv(seg, z=-1)
        ceil_idx = uv2idx(ceil_uv, 1024, 512)
        ceil_coordinates = torch.stack([ceil_idx[:, 1], ceil_idx[:, 0]])
        loss_ceilwall -= map_coordinates_Pytorch(scoreedg[..., 1], ceil_coordinates).mean() / len(segs)

    # floor-wall
    loss_floorwall = 0
    for seg in segs:
        floor_uv = xyz2uv(seg, z=pc_height)
        floor_idx = uv2idx(floor_uv, 1024, 512)
        floor_coordinates = torch.stack([floor_idx[:, 1], floor_idx[:, 0]])
        loss_floorwall -= map_coordinates_Pytorch(scoreedg[..., 2], floor_coordinates).mean() / len(segs)

    #losses = 1.0 * loss_cor + 0.1 * loss_wallwall + 0.5 * loss_ceilwall + 1.0 * loss_floorwall
    losses = 1.0 * loss_cor + 1.0 * loss_ceilwall + 1.0 * loss_floorwall

    # if i_step is not None:
    #     with torch.no_grad():
    #         print('step %d: %.3f (cor %.3f, wall %.3f, ceil %.3f, floor %.3f)' % (
    #             i_step, losses,
    #             loss_cor, loss_wallwall,
    #             loss_ceilwall, loss_floorwall))

    return losses


def vecang(vec1, vec2):
    vec1 = vec1 / np.sqrt((vec1 ** 2).sum())
    vec2 = vec2 / np.sqrt((vec2 ** 2).sum())
    return np.arccos(np.dot(vec1, vec2))


def rotatevec(vec, theta):
    x = vec[0] * torch.cos(theta) - vec[1] * torch.sin(theta)
    y = vec[0] * torch.sin(theta) + vec[1] * torch.cos(theta)
    return torch.cat([x, y])


def pts_linspace(pa, pb, pts=300):
    pa = pa.view(1, 2)
    pb = pb.view(1, 2)
    w = torch.arange(0, pts + 1, dtype=pa.dtype).view(-1, 1)
    return (pa * (pts - w) + pb * w) / pts


def xyz2uv(xy, z=-1):
    c = torch.sqrt((xy ** 2).sum(1))
    u = torch.atan2(xy[:, 1], xy[:, 0]).view(-1, 1)
    v = torch.atan2(torch.zeros_like(c) + z, c).view(-1, 1)
    return torch.cat([u, v], dim=1)


def uv2idx(uv, w, h):
    col = (uv[:, 0] / (2 * np.pi) + 0.5) * w - 0.5
    row = (uv[:, 1] / np.pi + 0.5) * h - 0.5
    return torch.cat([col.view(-1, 1), row.view(-1, 1)], dim=1)


def pc2cor_id(pc, pc_vec, pc_theta, pc_height):
    if pc_theta.numel() == 1:
        ps = torch.stack([
            (pc + pc_vec),
            (pc + rotatevec(pc_vec, pc_theta)),
            (pc - pc_vec),
            (pc + rotatevec(pc_vec, pc_theta - np.pi))
        ])
    else:
        ps = pc + pc_vec
        ps = ps.view(-1, 2)
        for c_num in range(pc_theta.shape[1]):
            ps = torch.cat((ps, ps[c_num:, :]), 0)
            if (c_num % 2) == 0:
                ps[-1, 1] = pc_theta[0, c_num]
            else:
                ps[-1, 0] = pc_theta[0, c_num]
        ps = torch.cat((ps, ps[-1:, :]), 0)
        ps[-1, 1] = ps[0, 1]

    return torch.cat([
        uv2idx(xyz2uv(ps, z=-1), 1024, 512),
        uv2idx(xyz2uv(ps, z=pc_height), 1024, 512),
    ], dim=0)
