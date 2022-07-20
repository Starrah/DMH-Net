import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from config import cfg, cfg_from_yaml_file, cfg_from_list
from e2plabel.e2plabelconvert import VIEW_NAME
from perspective_dataset import PerspectiveDataset
from visualization import getMaskByType, visualize
from postprocess.postprocess2 import get_vote_mask_c_up_down

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg_file', type=str, required=True, help='specify the config for training')
    parser.add_argument('--visu_count', default=2, type=int, help='visualize how many batches')
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    device = torch.device('cuda')

    dataset_valid = PerspectiveDataset(cfg, "test")
    loader_valid = DataLoader(dataset_valid,
                              args.batch_size,
                              collate_fn=dataset_valid.collate,
                              shuffle=False,
                              drop_last=False,
                              num_workers=0,
                              pin_memory=True)
    # 生成nearest_only类型的hough label的数据集
    dataset_nearest_only = PerspectiveDataset(cfg, "test")
    dataset_nearest_only.hough_label_gradual_type = "nearest_only"

    iterator_valid = iter(loader_valid)
    for valid_idx in trange(args.visu_count, desc='Verify CLine Vote', position=2):
        input = next(iterator_valid)

        def _core(input):
            with torch.no_grad():
                for k in input:
                    if isinstance(input[k], torch.Tensor):
                        input[k] = input[k].to(device)

                matss = []
                for img_idx in range(input["p_imgs"].shape[0]):
                    mats = []
                    for view_idx, view_name in enumerate(VIEW_NAME):
                        mat, _ = getMaskByType("gtc", cfg, input, None, img_idx, view_idx)
                        mats.append(mat)
                    matss.append(mats)
                gtc_map = torch.stack([torch.stack(mats, dim=0) for mats in matss], dim=0)
                vmask = get_vote_mask_c_up_down(cfg, input["p_imgs"])
                vmu, vmd = vmask[:, :, 0:vmask.shape[-1] // 2], vmask[:, :, vmask.shape[-1] // 2:]
                hough_c_up_vote = torch.matmul(gtc_map.reshape(*gtc_map.shape[0:2], -1), vmu.reshape(-1, vmu.shape[-1]))
                hough_c_down_vote = torch.matmul(gtc_map.reshape(*gtc_map.shape[0:2], -1),
                                                 vmd.reshape(-1, vmd.shape[-1]))
                hough_vote_res = torch.stack([hough_c_up_vote, hough_c_down_vote], dim=3)
                hough_vote_res = hough_vote_res / hough_vote_res.max()
                gtc_output = {
                    "raw_cud": hough_vote_res
                }

                visualize(cfg, input, gtc_output, drawtypes=[["c gt"], ["c raw"]], show=True, dpi=600)


        _core(input)

        _core(loader_valid.collate_fn([dataset_nearest_only.getItem(f) for f in input["filename"]]))
