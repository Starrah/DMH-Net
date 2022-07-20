import json

import open3d
from PIL import Image
from scipy.ndimage import map_coordinates
from tqdm import trange

from misc.panostretch import pano_connect_points
from misc.post_proc import np_coor2xy, np_coory2v


def xyz_2_coorxy(xs, ys, zs, H, W):
    ''' Mapping 3D xyz coordinates to equirect coordinate '''
    us = np.arctan2(xs, -ys)
    vs = -np.arctan(zs / np.sqrt(xs**2 + ys**2))
    coorx = (us / (2 * np.pi) + 0.5) * W
    coory = (vs / np.pi + 0.5) * H
    return coorx, coory


def create_ceiling_floor_mask(cor_id, H, W):
    '''
    Binary masking on equirectangular
    where 1 indicate floor or ceiling
    '''
    # Prepare 1d ceiling-wall/floor-wall boundary
    c_pts = []
    f_pts = []
    n_cor = len(cor_id)
    for i in range(n_cor // 2):
        # Ceiling boundary points
        xys = pano_connect_points(cor_id[i * 2],
                                  cor_id[(i * 2 + 2) % n_cor],
                                  z=-50,
                                  w=W,
                                  h=H)
        c_pts.extend(xys)

        # Floor boundary points
        xys = pano_connect_points(cor_id[i * 2 + 1],
                                  cor_id[(i * 2 + 3) % n_cor],
                                  z=50,
                                  w=W,
                                  h=H)
        f_pts.extend(xys)

    # Sort for interpolate
    c_pts = np.array(c_pts)
    c_pts = c_pts[np.argsort(c_pts[:, 0] * H - c_pts[:, 1])]
    f_pts = np.array(f_pts)
    f_pts = f_pts[np.argsort(f_pts[:, 0] * H + f_pts[:, 1])]

    # Removed duplicated point
    c_pts = np.concatenate([c_pts[:1], c_pts[1:][np.diff(c_pts[:, 0]) > 0]], 0)
    f_pts = np.concatenate([f_pts[:1], f_pts[1:][np.diff(f_pts[:, 0]) > 0]], 0)

    # Generate boundary for each image column
    c_bon = np.interp(np.arange(W), c_pts[:, 0], c_pts[:, 1])
    f_bon = np.interp(np.arange(W), f_pts[:, 0], f_pts[:, 1])

    # Generate mask
    mask = np.zeros((H, W), np.bool)
    for i in range(W):
        u = max(0, int(round(c_bon[i])) + 1)
        b = min(W, int(round(f_bon[i])))
        mask[:u, i] = 1
        mask[b:, i] = 1

    return mask


def warp_walls(equirect_texture, xy, floor_z, ceil_z, ppm):
    ''' Generate all walls' xyzrgba '''
    H, W = equirect_texture.shape[:2]
    all_rgb = []
    all_xyz = []
    for i in trange(len(xy), desc='Processing walls'):
        next_i = (i + 1) % len(xy)
        xy_a = xy[i]
        xy_b = xy[next_i]
        xy_w = np.sqrt(((xy_a - xy_b)**2).sum())
        t_h = int(round((ceil_z - floor_z) * ppm))
        t_w = int(round(xy_w * ppm))
        xs = np.linspace(xy_a[0], xy_b[0], t_w)[None].repeat(t_h, 0)
        ys = np.linspace(xy_a[1], xy_b[1], t_w)[None].repeat(t_h, 0)
        zs = np.linspace(floor_z, ceil_z, t_h)[:, None].repeat(t_w, 1)
        coorx, coory = xyz_2_coorxy(xs, ys, zs, H, W)

        plane_texture = np.stack([
            map_coordinates(equirect_texture[..., 0], [coory, coorx],
                            order=1,
                            mode='wrap'),
            map_coordinates(equirect_texture[..., 1], [coory, coorx],
                            order=1,
                            mode='wrap'),
            map_coordinates(equirect_texture[..., 2], [coory, coorx],
                            order=1,
                            mode='wrap'),
        ], -1)
        plane_xyz = np.stack([xs, ys, zs], axis=-1)

        all_rgb.extend(plane_texture.reshape(-1, 3))
        all_xyz.extend(plane_xyz.reshape(-1, 3))

    return all_rgb, all_xyz


def warp_floor_ceiling(equirect_texture, ceil_floor_mask, xy, z_floor,
                       z_ceiling, ppm):
    ''' Generate floor's and ceiling's xyzrgba '''
    assert equirect_texture.shape[:2] == ceil_floor_mask.shape[:2]
    H, W = equirect_texture.shape[:2]
    min_x = xy[:, 0].min()
    max_x = xy[:, 0].max()
    min_y = xy[:, 1].min()
    max_y = xy[:, 1].max()
    t_h = int(round((max_y - min_y) * ppm))
    t_w = int(round((max_x - min_x) * ppm))
    xs = np.linspace(min_x, max_x, t_w)[None].repeat(t_h, 0)
    ys = np.linspace(min_y, max_y, t_h)[:, None].repeat(t_w, 1)
    zs_floor = np.zeros_like(xs) + z_floor
    zs_ceil = np.zeros_like(xs) + z_ceiling
    coorx_floor, coory_floor = xyz_2_coorxy(xs, ys, zs_floor, H, W)
    coorx_ceil, coory_ceil = xyz_2_coorxy(xs, ys, zs_ceil, H, W)

    # Project view
    floor_texture = np.stack([
        map_coordinates(equirect_texture[..., 0], [coory_floor, coorx_floor],
                        order=1,
                        mode='wrap'),
        map_coordinates(equirect_texture[..., 1], [coory_floor, coorx_floor],
                        order=1,
                        mode='wrap'),
        map_coordinates(equirect_texture[..., 2], [coory_floor, coorx_floor],
                        order=1,
                        mode='wrap'),
    ], -1)
    floor_mask = map_coordinates(ceil_floor_mask, [coory_floor, coorx_floor],
                                 order=0)
    floor_xyz = np.stack([xs, ys, zs_floor], axis=-1)

    ceil_texture = np.stack([
        map_coordinates(equirect_texture[..., 0], [coory_ceil, coorx_ceil],
                        order=1,
                        mode='wrap'),
        map_coordinates(equirect_texture[..., 1], [coory_ceil, coorx_ceil],
                        order=1,
                        mode='wrap'),
        map_coordinates(equirect_texture[..., 2], [coory_ceil, coorx_ceil],
                        order=1,
                        mode='wrap'),
    ], -1)
    ceil_mask = map_coordinates(ceil_floor_mask, [coory_ceil, coorx_ceil],
                                order=0)
    ceil_xyz = np.stack([xs, ys, zs_ceil], axis=-1)

    floor_texture = floor_texture[floor_mask]
    floor_xyz = floor_xyz[floor_mask]
    ceil_texture = ceil_texture[ceil_mask]
    ceil_xyz = ceil_xyz[ceil_mask]

    return floor_texture, floor_xyz, ceil_texture, ceil_xyz


def create_occlusion_mask(xyz):
    xs, ys, zs = xyz.T
    ds = np.sqrt(xs**2 + ys**2 + zs**2)

    # Reorder by depth (from far to close)
    idx = np.argsort(-ds)
    xs, ys, zs, ds = xs[idx], ys[idx], zs[idx], ds[idx]

    # Compute coresponding quirect coordinate
    coorx, coory = xyz_2_coorxy(xs, ys, zs, H=256, W=512)
    quan_coorx = np.round(coorx).astype(int) % W
    quan_coory = np.round(coory).astype(int) % H

    # Generate layout depth
    depth_map = np.zeros((H, W), np.float32) + 1e9
    depth_map[quan_coory, quan_coorx] = ds
    tol_map = np.max([
        np.abs(np.diff(depth_map, axis=0, append=depth_map[[-2]])),
        np.abs(np.diff(depth_map, axis=1, append=depth_map[:, [0]])),
        np.abs(np.diff(depth_map, axis=1, prepend=depth_map[:, [-1]])),
    ], 0)

    # filter_ds = map_coordinates(depth_map, [coory, coorx], order=1, mode='wrap')
    # tol_ds = map_coordinates(tol_map, [coory, coorx], order=1, mode='wrap')
    filter_ds = depth_map[quan_coory, quan_coorx]
    tol_ds = tol_map[quan_coory, quan_coorx]
    mask = ds > (filter_ds + 2 * tol_ds)

    return mask, idx


"""Module which creates mesh lines from a line set
Open3D relies upon using glLineWidth to set line width on a LineSet
However, this method is now deprecated and not fully supporeted in newer OpenGL versions
See:
    Open3D Github Pull Request - https://github.com/intel-isl/Open3D/pull/738
    Other Framework Issues - https://github.com/openframeworks/openFrameworks/issues/3460

This module aims to solve this by converting a line into a triangular mesh (which has thickness)
The basic idea is to create a cylinder for each line segment, translate it, and then rotate it.

License: MIT

"""
import numpy as np
import open3d as o3d


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a),
                    center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img',
                        required=True,
                        help='Image texture in equirectangular format')
    parser.add_argument('--layout',
                        required=True,
                        help='Txt file containing layout corners (cor_id)')
    parser.add_argument('--camera_height',
                        default=1.6,
                        type=float,
                        help='Camera height in meter (not the viewer camera)')
    parser.add_argument('--ppm', default=80, type=int, help='Points per meter')
    parser.add_argument('--point_size',
                        default=0.0025,
                        type=int,
                        help='Point size')
    parser.add_argument('--ignore_floor',
                        action='store_true',
                        help='Skip rendering floor')
    parser.add_argument('--ignore_ceiling',
                        action='store_true',
                        help='Skip rendering ceiling')
    parser.add_argument('--ignore_wireframe',
                        action='store_true',
                        help='Skip rendering wireframe')
    args = parser.parse_args()

    # Reading source (texture img, cor_id txt)
    equirect_texture = np.array(Image.open(args.img)) / 255.0
    H, W = equirect_texture.shape[:2]
    with open(args.layout) as f:
        inferenced_result = json.load(f)
    cor_id = np.array(inferenced_result['uv'], np.float32)
    cor_id[:, 0] *= W
    cor_id[:, 1] *= H

    ceil_floor_mask = create_ceiling_floor_mask(cor_id, H, W)

    # Convert cor_id to 3d xyz
    N = len(cor_id) // 2
    floor_z = -args.camera_height
    floor_xy = np_coor2xy(cor_id[1::2], floor_z, W, H, floorW=1, floorH=1)
    c = np.sqrt((floor_xy**2).sum(1))
    v = np_coory2v(cor_id[0::2, 1], H)
    ceil_z = (c * np.tan(v)).mean()

    # Prepare
    if not args.ignore_wireframe:
        assert N == len(floor_xy)
        wf_points = [[-x, y, floor_z] for x, y in floor_xy] +\
                    [[-x, y, ceil_z] for x, y in floor_xy]
        wf_lines = [[i, (i+1)%N] for i in range(N)] +\
                   [[i+N, (i+1)%N+N] for i in range(N)] +\
                   [[i, i+N] for i in range(N)]
        wf_colors = [[0, 1, 0] if i % 2 == 0 else [0, 0, 1] for i in range(N)] +\
                    [[0, 1, 0] if i % 2 == 0 else [0, 0, 1] for i in range(N)] +\
                    [[1, 0, 0] for i in range(N)]
        wf_line_set = open3d.geometry.LineSet()
        wf_line_set.points = open3d.utility.Vector3dVector(wf_points)
        wf_line_set.lines = open3d.utility.Vector2iVector(wf_lines)
        wf_line_set.colors = open3d.utility.Vector3dVector(wf_colors)

    # Warp each wall
    all_rgb, all_xyz = warp_walls(equirect_texture, floor_xy, floor_z, ceil_z,
                                  args.ppm)

    # Warp floor and ceiling
    if not args.ignore_floor or not args.ignore_ceiling:
        fi, fp, ci, cp = warp_floor_ceiling(equirect_texture,
                                            ceil_floor_mask,
                                            floor_xy,
                                            floor_z,
                                            ceil_z,
                                            ppm=args.ppm)

        if not args.ignore_floor:
            all_rgb.extend(fi)
            all_xyz.extend(fp)

        if not args.ignore_ceiling:
            all_rgb.extend(ci)
            all_xyz.extend(cp)

    all_xyz = np.array(all_xyz)
    all_rgb = np.array(all_rgb)

    all_xyz = all_xyz * np.array([-1,1,1])

    # Filter occluded points
    occlusion_mask, reord_idx = create_occlusion_mask(all_xyz)
    all_xyz = all_xyz[reord_idx][~occlusion_mask]
    all_rgb = all_rgb[reord_idx][~occlusion_mask]

    # Launch point cloud viewer
    print('Showing %d of points...' % len(all_rgb))
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(all_xyz)
    pcd.colors = open3d.utility.Vector3dVector(all_rgb)

    # Visualize result
    tobe_visualize = [pcd]
    if not args.ignore_wireframe:
        # tobe_visualize.append(wf_line_set)
        line_mesh1 = LineMesh(wf_points, wf_lines, wf_colors, radius=0.04)
        line_mesh1_geoms = line_mesh1.cylinder_segments
        tobe_visualize.extend(line_mesh1_geoms)

    open3d.visualization.RenderOption.line_width = 10.0
    open3d.visualization.draw_geometries(tobe_visualize)

