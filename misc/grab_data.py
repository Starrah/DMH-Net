'''
Test docstring
'''
# pylint: disable=cell-var-from-loop
# pylint --extension-pkg-whitelist=cv2
import glob
import urllib.request
import json
import math
import numpy as np
import cv2
from tqdm import trange
import argparse
import os


def dump_to_txt(m_path, m_list):
    '''Dump a list of string to the given m_path.'''
    with open(m_path, 'w') as f:
        for dtt_idx in range(m_list.shape[0]):
            f.write("%d %d\n" % (m_list[dtt_idx, 0], m_list[dtt_idx, 1]))


def process_mark_points(pmp_obj, pmp_rows, pmp_cols):
    '''Process obj (such as door, wall, etc.) to give their 3d coords'''
    sux = pmp_obj['StartMarkPoint']['UpPosition']['x']
    suz = pmp_obj['StartMarkPoint']['UpPosition']['z']
    suy = pmp_obj['StartMarkPoint']['UpPosition']['y']
    sdy = pmp_obj['StartMarkPoint']['DownPosition']['y']
    azimuth = math.atan2(sux, -suz)
    elevation_u = math.atan(suy / np.sqrt(suz * suz + sux * sux))
    elevation_d = math.atan(sdy / np.sqrt(suz * suz + sux * sux))
    pmp_col0 = int(pmp_cols - (azimuth * 0.5 / np.pi + 0.5) * pmp_cols)
    pmp_row0_u = int(pmp_rows / 2 - elevation_u / np.pi * pmp_rows)
    pmp_row0_d = int(pmp_rows / 2 - elevation_d / np.pi * pmp_rows)

    eux = pmp_obj['EndMarkPoint']['UpPosition']['x']
    euz = pmp_obj['EndMarkPoint']['UpPosition']['z']
    euy = pmp_obj['EndMarkPoint']['UpPosition']['y']
    edy = pmp_obj['EndMarkPoint']['DownPosition']['y']
    azimuth = math.atan2(eux, -euz)
    elevation_u = math.atan(euy / np.sqrt(euz * euz + eux * eux))
    elevation_d = math.atan(edy / np.sqrt(euz * euz + eux * eux))
    pmp_col1 = int(pmp_cols - (azimuth * 0.5 / np.pi + 0.5) * pmp_cols)
    pmp_row1_u = int(pmp_rows / 2 - elevation_u / np.pi * pmp_rows)
    pmp_row1_d = int(pmp_rows / 2 - elevation_d / np.pi * pmp_rows)
    return pmp_col0, pmp_col1, pmp_row0_u, pmp_row0_d, pmp_row1_u, pmp_row1_d


# file_path = '/data00/xuezhou/datasets/grabData/woaiwojia_vr.txt'
# f = open(file_path, 'r')
# for line in f:
#     file_name = os.path.basename(line)
#     save_as = '/data00/xuezhou/datasets/grabData/jsonList/%s.json'%(file_name[:-1])
#     print(save_as)
#     urllib.request.urlretrieve(line, save_as)


def grab_pano_urls():
    json_list = glob.glob('/data00/xuezhou/datasets/grabData/jsonList/*.json')
    test_urls = []
    for json_path in json_list:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        for floor in json_data['Floors']:
            Rooms = floor['Rooms']
            HotSpots = json_data['HotSpots']
            for room in Rooms:
                room_ids = room['HotSpotIds']
                for room_id in room_ids:
                    matchingHotSpot = list(filter(lambda hs_ele: hs_ele['ID'] == room_id, HotSpots))
                    if not matchingHotSpot:
                        print('no matching for room %s from' % room_id, 'in', json_path)
                        continue
                    tos_url = matchingHotSpot[0]['tos_origin_url']
                    if not tos_url:
                        print('no valid tos_origin_url for room %s in' % room_id, 'in', json_path)
                        continue
                    test_urls.append(tos_url)
        if (len(test_urls) > 1000):
            break
    with open('./test_urls.json', 'w') as f:
        json.dump(test_urls, f)


if __name__ == '__main__':
    # Set all parameters for grab data
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--start_idx', default=-1, type=int, help='starting index of the sorted json list')
    parser.add_argument('--end_idx', default=-1, type=int, help='ending index of the sorted json list')
    args = parser.parse_args()

    SAVE_DATA_FOLDER = '/mnt/cephfs_new_wj/uslabcv/wenchao/datasets/wawj_ori_datasets/wen_grab'
    JSON_LIST = sorted(
        glob.glob('/mnt/cephfs_new_wj/uslabcv/wenchao/datasets/wawj_ori_datasets/grabData/jsonListMultiFloor/*.json'))
    SUB_FOLDER = 'test'
    ROWS, COLS = 512, 1024

    START_INDEX = 0 if (args.start_idx < 0) else args.start_idx
    END_INDEX = len(JSON_LIST) if (args.end_idx < 0) else args.end_idx

    label_dir = os.path.join(SAVE_DATA_FOLDER, SUB_FOLDER, 'label_cor')
    img_dir = os.path.join(SAVE_DATA_FOLDER, SUB_FOLDER, 'img')
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # Grab data
    for ith_json in trange(START_INDEX, END_INDEX, desc='Grabbing', unit='file'):
        # Load and parse json file
        json_path = JSON_LIST[ith_json]
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        HotSpots = json_data['HotSpots']
        hotspots_ids_map = {HotSpots[k]['ID']: HotSpots[k] for k in range(len(HotSpots))}

        for floor in json_data['Floors']:
            Rooms = floor['Rooms']
            for room in Rooms:
                walls = room['Walls']
                if (len(walls) < 4):
                    continue
                doors = room['Doors']
                windows = room['Windows']

                room_ids = room['HotSpotIds']
                # import ipdb; ipdb.set_trace()
                for room_id in room_ids:  # TODO: why list?
                    # check if room_id is valid
                    if room_id not in hotspots_ids_map:
                        print('no matching for room %s from' % room_id, ith_json, 'json file')
                        continue

                    matchingHotSpot = hotspots_ids_map[room_id]
                    tos_url = matchingHotSpot['tos_origin_url']
                    if not tos_url:
                        print('no valid tos_origin_url for room %s in' % room_id, ith_json, 'json file')
                        continue
                    url_name = os.path.basename(tos_url)
                    file_name = '%05d_%s' % (ith_json, url_name)
                    save_as = os.path.join(img_dir, '%s.png' % file_name)

                    # processing wall
                    gt_point = []
                    pre_col1 = -1
                    for wall in walls:
                        col0, col1, row0_u, row0_d, row1_u, row1_d = process_mark_points(wall, ROWS, COLS)
                        if (pre_col1 < 0):
                            pre_col1 = col1
                        else:
                            if (pre_col1 == col0):
                                pre_col1 = col1
                            else:
                                pre_col1 = -1
                                break
                        gt_point.append([col0, row0_u])
                        gt_point.append([col0, row0_d])
                    if ((pre_col1 == -1) or (col1 != gt_point[0][0])):
                        continue
                    gt_point = np.array(gt_point)

                    # processing door
                    door_seg_point = []
                    for door in doors:
                        col0, col1, row0_u, row0_d, row1_u, row1_d = process_mark_points(door, ROWS, COLS)
                        door_seg_point.append([col0, row0_u])
                        door_seg_point.append([col0, row0_d])
                        door_seg_point.append([col1, row1_u])
                        door_seg_point.append([col1, row1_d])
                    door_seg_point = np.array(door_seg_point)

                    # processing window
                    window_seg_point = []
                    for window in windows:
                        col0, col1, row0_u, row0_d, row1_u, row1_d = process_mark_points(window, ROWS, COLS)
                        window_seg_point.append([col0, row0_u])
                        window_seg_point.append([col0, row0_d])
                        window_seg_point.append([col1, row1_u])
                        window_seg_point.append([col1, row1_d])
                    window_seg_point = np.array(window_seg_point)

                    wall_label_path = os.path.join(label_dir, '%s.txt' % file_name)
                    dump_to_txt(wall_label_path, gt_point)

                    door_label_path = os.path.join(label_dir, '%s_ds.txt' % file_name)
                    dump_to_txt(door_label_path, door_seg_point)

                    window_label_path = os.path.join(label_dir, '%s_ws.txt' % file_name)
                    dump_to_txt(window_label_path, window_seg_point)

                    urllib.request.urlretrieve(tos_url, save_as)
                    pano_image = cv2.imread(save_as)
                    cv2.imwrite(save_as, cv2.resize(pano_image, (COLS, ROWS)))
                    break
