'''
This script preprocess the given 360 panorama image under euqirectangular projection
and dump them to the given directory for further layout prediction and visualization.
The script will:
    - extract and dump the vanishing points
    - rotate the equirect image to align with the detected VP
    - extract the VP aligned line segments (for further layout prediction model)
The dump files:
    - `*_VP.txt` is the vanishg points
    - `*_aligned_rgb.png` is the VP aligned RGB image
    - `*_aligned_line.png` is the VP aligned line segments images

Author: Cheng Sun
Email : chengsun@gapp.nthu.edu.tw
'''

import os
import glob
import argparse
import numpy as np
from PIL import Image
import sys
from misc.pano_lsd_align import panoEdgeDetection, rotatePanorama, uv2xyzN, xyz2uvN
from misc.grab_data import dump_to_txt


def alignLabel(cor_points, vp_data, sphereH, sphereW):
    TX = cor_points[:, 0].reshape(-1, 1) + 1
    TY = cor_points[:, 1].reshape(-1, 1) + 1
    ANGx = (TX - sphereW / 2 - 0.5) / sphereW * np.pi * 2
    ANGy = -(TY - sphereH / 2 - 0.5) / sphereH * np.pi
    uvOld = np.hstack([ANGx, ANGy])
    xyzOld = uv2xyzN(uvOld, 1)

    R = np.linalg.inv(vp_data.T)
    xyzNew = xyzOld @ R.T
    uvNew = xyz2uvN(xyzNew, 1)
    Px = (uvNew[:, 0] + np.pi) / (2 * np.pi) * sphereW + 0.5
    Py = (-uvNew[:, 1] + np.pi / 2) / np.pi * sphereH + 0.5
    gt_points = np.vstack([Px, Py]).astype(int).T
    return gt_points


def saveRelatedLabel(base_folder, base_name, vp_data, sphereH, sphereW):
    '''
    After algin the pano image, we should also algin the gt cor and door
    '''
    cor_path = '%s/../label_cor/%s.txt' % (base_folder, base_name)
    ds_path = '%s/../label_cor/%s_ds.txt' % (base_folder, base_name)
    if not (os.path.isfile(os.path.join(cor_path))
            and os.path.isfile(os.path.join(ds_path))):
        print('fail to locate %s and %s' % (cor_path, ds_path))
        return False

    with open(cor_path) as f:
        cor = np.array([line.strip().split() for line in f if line.strip()],
                       np.float32)
    if not cor.any():
        return False
    gt_points = alignLabel(cor, vp_data, sphereH, sphereW)
    dump_to_txt('%s/../label_cor_bak/%s.txt' % (base_folder, base_name),
                gt_points)

    with open(ds_path) as f:
        cor_ds = np.array([line.strip().split() for line in f if line.strip()],
                          np.float32)
    if not cor_ds.any():
        return False
    ds_points = alignLabel(cor_ds, vp_data, sphereH, sphereW)
    dump_to_txt('%s/../label_cor_bak/%s_ds.txt' % (base_folder, base_name),
                ds_points)

    return True


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter)
# I/O related arguments
parser.add_argument('--img_glob',
                    required=True,
                    help='NOTE: Remeber to quote your glob path.')
parser.add_argument('--output_dir', required=True)
parser.add_argument('--rgbonly',
                    action='store_true',
                    help='Add this if use are preparing customer dataset')
# Preprocessing related arguments
parser.add_argument('--q_error', default=0.7, type=float)
parser.add_argument('--refine_iter', default=3, type=int)
parser.add_argument('--start_idx', default=-1, type=int)
parser.add_argument('--end_idx', default=-1, type=int)
args = parser.parse_args()

paths = sorted(glob.glob(args.img_glob))
if len(paths) == 0:
    sys.exit('no images found')

# Check given path exist
for path in paths:
    assert os.path.isfile(path), '%s not found' % path

# Check target directory
if not os.path.isdir(args.output_dir):
    print('Output directory %s not existed. Create one.')
    os.makedirs(args.output_dir)

START_INDEX = 0 if (args.start_idx < 0) else args.start_idx
END_INDEX = len(paths) if (args.end_idx < 0) else args.end_idx
file_list = list(
    filter(lambda x: os.path.basename(x).startswith('%05d_' % START_INDEX),
           paths))
if not file_list:
    sys.exit('Wrong starting idx for file')
start_list_idx = paths.index(file_list[0])

END_INDEX = 1 #len(paths) if (END_INDEX > len(paths)) else END_INDEX
file_list = list(
    filter(lambda x: os.path.basename(x).startswith('%05d_' % END_INDEX),
           paths))
#if not file_list:
#    sys.exit('Wrong ending idx for file')
#end_list_idx = paths.index(file_list[0])

# Process each input
#for i_path in paths:
for path_index in range(len(paths)): #(start_list_idx, end_list_idx):
    i_path = paths[path_index]
    print('Processing', i_path, flush=True)

    # Load and cat input images
    img_ori = np.array(Image.open(i_path).resize((1024, 512),
                                                 Image.BICUBIC))[..., :3]

    # VP detection and line segment extraction
    _, vp, _, _, panoEdge, _, _ = panoEdgeDetection(
        img_ori, qError=args.q_error, refineIter=args.refine_iter)
    panoEdge = (panoEdge > 0)

    # Align images with VP
    i_img = rotatePanorama(img_ori / 255.0, vp[2::-1])
    l_img = rotatePanorama(panoEdge.astype(np.float32), vp[2::-1])

    # Dump results
    basename = os.path.splitext(os.path.basename(i_path))[0]
    if args.rgbonly:
        path = os.path.join(args.output_dir, '%s.png' % basename)
        path_to_label_cor = '%s/../label_cor/%s.txt'
        path_to_label_ds = '%s/../label_cor/%s_ds.txt'
        if (saveRelatedLabel(args.output_dir, basename, vp[2::-1], 512, 1024)):
            Image.fromarray((i_img * 255).astype(np.uint8)).save(path)
        else:
            print('Failed to process %s', path)
    else:
        path_VP = os.path.join(args.output_dir, '%s_VP.txt' % basename)
        path_i_img = os.path.join(args.output_dir,
                                  '%s_aligned_rgb.png' % basename)
        path_l_img = os.path.join(args.output_dir,
                                  '%s_aligned_line.png' % basename)

        with open(path_VP, 'w') as f:
            for i in range(3):
                f.write('%.6f %.6f %.6f\n' % (vp[i, 0], vp[i, 1], vp[i, 2]))
        Image.fromarray((i_img * 255).astype(np.uint8)).save(path_i_img)
        Image.fromarray((l_img * 255).astype(np.uint8)).save(path_l_img)
