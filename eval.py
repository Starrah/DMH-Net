import argparse
import json
import os
# import ipdb
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import trange

from config import cfg, cfg_from_yaml_file, cfg_from_list, merge_new_config
from misc.utils import pipeload
from model import ENCODER_RESNET, ENCODER_DENSENET, ENCODER_HOUGH, DMHNet
from perspective_dataset import PerspectiveDataset, worker_init_fn
from postprocess.postprocess2 import postProcess
from visualization import visualizeWithPostResults

from torch.nn import functional as F

SAVE_JSON = False


def valid(cfg, net, loader_valid, dataset_valid, device, visualize_count=0, show=False, dpi=None, print_detail=False,
          valid_epoch=None):
    """
    验证用的核心函数
    :param valid_epoch None表示是test，数字表示是valid，值表示触发valid的epoch number
    """
    net.eval()
    # torch.cuda.empty_cache()
    iterator_valid = iter(loader_valid)
    valid_loss = {}
    metrics = {}
    metrics_by_corner = {}
    imgs = {}
    visualize_index = np.zeros(len(loader_valid), dtype=bool)
    visualize_index[np.random.choice(len(loader_valid), size=visualize_count, replace=False)] = True
    for valid_idx in trange(len(loader_valid), desc='Eval', position=2):
        input = next(iterator_valid)
        valid_batch_size = input["e_img"].size(0)
        with torch.no_grad():
            for k in input:
                if isinstance(input[k], torch.Tensor):
                    input[k] = input[k].to(device)
            losses, results_dict = net(input)

            postResults = []
            for i in range(len(input["filename"])):
                print(input["filename"][i])
                postStartTime = time.time()
                postResult = postProcess(cfg, input, results_dict, i, is_valid_mode=valid_epoch is not None)
                postResults.append(postResult)
                if print_detail:
                    (_, gt_lwh, _), (_, pred_lwh, _), metric = postResult
                    print("{:s} pred{:s} gt{:s} {:s}".format(str(metric), str(pred_lwh), str(gt_lwh),
                                                             input["filename"][i]))
                if SAVE_JSON or ("args" in globals() and args.print_json):
                    (_, _, _), (_, _, pred_cors), metric = postResult
                    uv = pred_cors.cpu().numpy() / input["e_img"].shape[-1:-3:-1]
                    uv = [[o.item() for o in pt] for pt in uv]
                    if SAVE_JSON:
                        JSON_DIR = "./result_json"
                        os.makedirs(JSON_DIR, exist_ok=True)
                        with open(os.path.join(JSON_DIR, input["filename"][i] + ".json"), "w") as f:
                            json.dump({"uv": uv, "3DIoU": metric["3DIoU"].item()}, f)
                    elif "args" in globals() and args.print_json:
                        print(json.dumps({"uv": uv, "3DIoU": metric["3DIoU"].item()}))

                _, _, metric = postResult
                for k, v in metric.items():
                    if isinstance(v, str): continue
                    metrics[k] = metrics.get(k, 0) + v.item()
                    if "n_corners_type" in metric:
                        k2 = metric["n_corners_type"] + "/" + k
                        if k2 not in metrics_by_corner: metrics_by_corner[k2] = []
                        metrics_by_corner[k2].append(v.item())
                metrics["gt_n_corners"] = metrics.get("gt_n_corners", 0) + (len(input["cor"][i]) // 2)

            for k, v in losses.items():
                valid_loss[k] = valid_loss.get(k, 0) + v.item() * valid_batch_size

            if visualize_index[valid_idx]:
                visualize_type = cfg.get("VISUALIZATION", {}).get("TYPE")
                imgs.update(visualizeWithPostResults(cfg, input, results_dict, postResults, drawtypes=visualize_type,
                                                     show=show, dpi=dpi))

    for k, v in valid_loss.items():
        valid_loss[k] = v / len(dataset_valid)

    for k, v in metrics.items():
        metrics[k] = v / len(dataset_valid)

    for k, v in metrics_by_corner.items():
        metrics[k] = torch.tensor(v).mean().item()

    return valid_loss, imgs, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg_file', type=str, required=True, help='specify the config for training')
    parser.add_argument('--ckpt', required=True, help='checkpoint for evaluation')

    parser.add_argument('--visu_count', default=0, type=int, help='visualize how many batches')
    parser.add_argument('--visu_all', action='store_true', help='visualize all samples')
    parser.add_argument('--visu_path', help='where to save the visualization result (default: plt.show)')
    parser.add_argument('--visu_type',
                        help='specify visualization type (either str or List[str], see visualization.py)')
    parser.add_argument('--no_post_process', action='store_true', help='don\'t post process')
    parser.add_argument('--develop_post_process', action='store_true', help='use POST_PROCESS.METHOD = \'develop\'')
    parser.add_argument('--valid_set', action='store_true', help='use valid set')

    parser.add_argument('--batch_size', default=2, type=int, help='mini-batch size')
    parser.add_argument('--input_file', type=str, help='eval on one single input image')
    parser.add_argument('--print_detail', action='store_true', help='print detail for each sample')
    parser.add_argument('--save_json', action='store_true', help='save json to ./result_json')
    parser.add_argument('--print_json', action='store_true', help='print json for each sample')
    parser.add_argument('--output_file', nargs="?", const=True,
                        help='whether to output to file 如果不填写参数，默认输出到eval_outputs/{time}.out')

    # Model related
    parser.add_argument('--backbone',
                        default='drn38',
                        choices=ENCODER_RESNET + ENCODER_DENSENET + ENCODER_HOUGH,
                        help='backbone of the network')
    parser.add_argument('--no_rnn', action='store_true', help='whether to remove rnn or not')
    # Dataset related arguments
    # TODO 原始代码交换了测试集与训练集 没有验证集
    # 新代码用的就是原始的训练集和测试集
    # parser.add_argument('--train_root_dir',
    #                     default='data/layoutnet_dataset/test',
    #                     help='root directory to training dataset. '
    #                          'should contains img, label_cor subdirectories')
    parser.add_argument('--valid_root_dir',
                        default='data/layoutnet_dataset/train',
                        help='root directory to validation dataset. '
                             'should contains img, label_cor subdirectories')
    parser.add_argument('--num_workers', default=4 if not sys.gettrace() else 0, type=int,
                        help='numbers of workers for dataloaders')
    # Misc arguments
    parser.add_argument('--no_cuda', action='store_true', help='disable cuda')
    parser.add_argument('--seed', default=594277, type=int, help='manual seed')
    parser.add_argument('--disp_iter', type=int, default=1, help='iterations frequency to display')
    parser.add_argument('--no_multigpus', action='store_true', help='disable data parallel')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    args = parser.parse_args()

    if args.save_json: SAVE_JSON = True

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    output_file = args.output_file if args.output_file != True else "eval_outputs/{:d}.out".format(int(time.time()))
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        output_file = open(output_file, "w")

    if args.visu_type:
        merge_new_config(cfg, {"VISUALIZATION": {"TYPE": yaml.safe_load(args.visu_type)}})

    if args.no_post_process:
        cfg.POST_PROCESS.METHOD = "None"
    elif args.develop_post_process:
        cfg.POST_PROCESS.METHOD = "develop"

    device = torch.device('cpu' if args.no_cuda else 'cuda')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    result_dir = os.path.join("eval_result", str(int(time.time())))
    os.makedirs(result_dir, exist_ok=True)

    # Create dataloader
    print("num_workers: " + str(args.num_workers))

    dataset_valid = PerspectiveDataset(cfg, "test" if not args.valid_set else "valid",  # TODO 新代码现在是用测试集进行验证的
                                       filename=args.input_file)
    loader_valid = DataLoader(dataset_valid,
                              args.batch_size,
                              collate_fn=dataset_valid.collate,
                              shuffle=False,
                              drop_last=False,
                              num_workers=args.num_workers,
                              pin_memory=not args.no_cuda,
                              worker_init_fn=worker_init_fn)

    # Create model
    net = DMHNet(cfg, cfg.MODEL.get("BACKBONE", {}).get("NAME", "drn38"), not args.no_rnn).to(device)

    if not args.no_multigpus:
        net = nn.DataParallel(net)  # multi-GPU

    print(str(cfg.POST_PROCESS))
    if output_file: output_file.write(str(cfg.POST_PROCESS) + "\n\n")

    if args.ckpt == "None":
        warnings.warn("ckpt参数显式传入了None！将不会加载任何参数！")
    else:
        state_dict = pipeload(args.ckpt, map_location='cpu')["state_dict"]
        net.load_state_dict(state_dict, strict=True)

    visualize_count = len(loader_valid) if args.visu_all else args.visu_count
    show = args.visu_path is None
    valid_loss, imgs, metrics = valid(cfg, net, loader_valid, dataset_valid, device, visualize_count, show=show,
                                      dpi=200, print_detail=args.print_detail)

    for k, v in valid_loss.items():
        k = 'eval_loss/%s' % k
        print("{:s} {:f}".format(k, v))
        if output_file: output_file.write("{:s} {:f}".format(k, v) + "\n")

    for k, v in metrics.items():
        k = 'metric/%s' % k
        print("{:s} {:f}".format(k, v))
        if output_file: output_file.write("{:s} {:f}".format(k, v) + "\n")

    if output_file:
        output_file.write("\n\n")
        output_file.write(str(cfg) + "\n")

    for k, v in imgs.items():
        if args.visu_path:
            os.makedirs(args.visu_path, exist_ok=True)
            success = cv2.imwrite(os.path.join(args.visu_path, k + ".jpg"), cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
            assert success, "write output image fail!"
