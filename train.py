import argparse
import os
# import ipdb
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import trange

from config import cfg, cfg_from_yaml_file, cfg_from_list
from eval import valid
from misc.utils import save_model, load_trained_model
from model import ENCODER_RESNET, ENCODER_DENSENET, ENCODER_HOUGH, DMHNet
from perspective_dataset import PerspectiveDataset, worker_init_fn

GAMMA = 2
ALPHA_XY = 1.0
ALPHA_MATCH = 10.0
ALPHA_ANGLE = 1.0
ALPHA_HEIGHT = 1.0


def feed_forward(net, x, angle, up_bins, down_bins, edge, height, return_results=False):
    up_bin256 = up_bins
    down_bin256 = down_bins
    x = x.to(device)
    angle = angle.to(device)
    up_bin256 = up_bin256.to(device)
    down_bin256 = down_bin256.to(device)
    edge = edge.to(device)
    height = height.to(device)

    losses = {}
    angle_, up_xy_, down_xy_, edge_, height_, results_dict = net(x)

    # Match loss
    # Edge classification loss
    loss_edg = F.binary_cross_entropy_with_logits(edge_, edge, reduction='none')
    loss_edg[edge == 0.] *= 0.2
    loss_edg = loss_edg.mean()
    losses['edge'] = loss_edg
    # Height loss
    losses['height'] = ALPHA_HEIGHT * F.l1_loss(height_, height)
    # X-Y classification loss
    # losses['fuse_xy'] = ALPHA_XY * F.binary_cross_entropy_with_logits(fuse_xy_, up_bin256)

    losses['up_xy'] = ALPHA_XY * F.binary_cross_entropy_with_logits(up_xy_, up_bin256)
    losses['down_xy'] = ALPHA_XY * F.binary_cross_entropy_with_logits(down_xy_, down_bin256)
    # Angle classification loss
    loss_cor_ori = ALPHA_ANGLE * F.binary_cross_entropy_with_logits(angle_, angle)
    # pt_cor = torch.exp(-loss_cor_ori)
    losses['angle'] = loss_cor_ori
    # ALPHA_ANGLE * ((1 - pt_cor)**GAMMA * loss_cor_ori).mean()

    idx = torch.arange(256).view(1, 256, 1)
    idx = idx.to(device)
    up_reg = (idx * F.softmax(up_xy_, 2)).sum(2).squeeze(1)
    down_reg = (idx * F.softmax(down_xy_, 2)).sum(2).squeeze(1)

    ratio = up_reg / (down_reg + 1e-8)

    losses['match'] = torch.abs(ratio - 1.).mean()

    # Total loss
    losses['total'] = losses['up_xy'] + losses['down_xy'] + losses['angle'] + losses['edge']
    losses['total'] += losses['height']
    losses['total'] += losses['match']
    # For model selection
    with torch.no_grad():
        nobrain_baseline_xy = 1.
        score_xy_up = 1 - (torch.sigmoid(up_xy_) - up_bin256).abs().mean() / nobrain_baseline_xy
        score_xy_down = 1 - (torch.sigmoid(down_xy_) - down_bin256).abs().mean() / nobrain_baseline_xy
        nobrain_baseline_angle = 1.
        score_angle = 1 - (torch.sigmoid(angle_) - angle).abs().mean() / nobrain_baseline_angle
        losses['score'] = (score_angle + score_xy_up + score_xy_down) / 3

    results_dict['angle'] = angle_.detach()
    results_dict['up_xy'] = up_xy_.detach()
    results_dict['down_xy'] = down_xy_.detach()

    if return_results:
        return losses, results_dict
    else:
        return losses


def feature_viz(name, tb_writer):
    def hook(model, input, output):
        feat = output.detach()
        feat_reshape = feat.view(-1, 1, feat.shape[2], feat.shape[3])
        img = make_grid(feat_reshape, normalize=True)
        tb_writer.add_image(name, img.cpu())

    return hook


def visualize_item(x, y_cor, results_dict):
    x = (x.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
    y_cor = y_cor.numpy()
    gt_cor = np.zeros((30, 1024, 3), np.uint8)
    gt_cor[:] = y_cor[0][None, :, None] * 255
    img_pad = np.zeros((3, 1024, 3), np.uint8) + 255
    cor_img = np.concatenate([gt_cor, img_pad, x], 0)

    up_img = results_dict['up_img'].detach().cpu()[0]
    up_img = (up_img.clone().numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
    down_img = results_dict['down_img'].detach().cpu()[0]
    down_img = (down_img.clone().numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)

    xy = torch.sigmoid(results_dict['up_xy']).detach().cpu()[0, 0].clone().numpy()
    dir_x_up = np.concatenate([xy[:, 0][::-1], xy[:, 2]], 0)
    dir_y_up = np.concatenate([xy[:, 1][::-1], xy[:, 3]], 0)
    x_up_prob = np.zeros((30, 512, 3), np.uint8)
    x_up_prob[:] = dir_x_up[None, :, None] * 255
    y_up_prob = np.zeros((512, 30, 3), np.uint8)
    y_up_prob[:] = dir_y_up[:, None, None] * 255
    stich_up_canvas = np.zeros((30 + 3 + 512, 30 + 3 + 512, 3), np.uint8) + 255
    stich_up_canvas[33:, 33:, :] = up_img
    stich_up_canvas[33:, :30, :] = y_up_prob
    stich_up_canvas[:30, 33:, :] = x_up_prob

    xy = torch.sigmoid(results_dict['down_xy']).detach().cpu()[0, 0].clone().numpy()
    dir_x_down = np.concatenate([xy[:, 0][::-1], xy[:, 2]], 0)
    dir_y_down = np.concatenate([xy[:, 1][::-1], xy[:, 3]], 0)
    x_down_prob = np.zeros((30, 512, 3), np.uint8)
    x_down_prob[:] = dir_x_down[None, :, None] * 255
    y_down_prob = np.zeros((512, 30, 3), np.uint8)
    y_down_prob[:] = dir_y_down[:, None, None] * 255
    stich_down_canvas = np.zeros((30 + 3 + 512, 30 + 3 + 512, 3), np.uint8) + 255
    stich_down_canvas[33:, 33:, :] = down_img
    stich_down_canvas[33:, :30, :] = y_down_prob
    stich_down_canvas[:30, 33:, :] = x_down_prob

    return cor_img, stich_up_canvas, stich_down_canvas


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg_file', '-c', type=str, required=True, help='specify the config for training')

    parser.add_argument('--id', required=True, help='experiment id to name checkpoints and logs')
    parser.add_argument('--ckpt', default='./ckpt', help='folder to output checkpoints')
    parser.add_argument('--logs', default='./logs', help='folder to logging')
    parser.add_argument('--pth', default=None, help='path to load saved checkpoint.' '(finetuning)')
    # Model related
    parser.add_argument('--backbone',
                        default='drn38',
                        choices=ENCODER_RESNET + ENCODER_DENSENET + ENCODER_HOUGH,
                        help='backbone of the network')
    parser.add_argument('--no_rnn', action='store_true', help='whether to remove rnn or not')
    # Dataset related arguments
    # TODO 原始代码交换了测试集与训练集 没有验证集
    # 新代码用的就是原始的训练集和测试集
    parser.add_argument('--train_root_dir',
                        default='data/layoutnet_dataset/test',
                        help='root directory to training dataset. '
                             'should contains img, label_cor subdirectories')
    parser.add_argument('--valid_root_dir',
                        default='data/layoutnet_dataset/train',
                        help='root directory to validation dataset. '
                             'should contains img, label_cor subdirectories')
    parser.add_argument('--no_flip', action='store_true', help='disable left-right flip augmentation')
    parser.add_argument('--no_rotate', action='store_true', help='disable horizontal rotate augmentation')
    parser.add_argument('--no_gamma', action='store_true', help='disable gamma augmentation')
    parser.add_argument('--no_erase', action='store_true', help='disable radom erasing augmentation')
    parser.add_argument('--no_noise', action='store_true', help='disable radom noise augmentation')
    parser.add_argument('--no_pano_stretch', action='store_true', help='disable pano stretch')

    parser.add_argument('--num_workers', '-j', type=int, help='numbers of workers for dataloaders')
    # optimization related arguments
    parser.add_argument('--freeze_earlier_blocks', default=-1, type=int)
    parser.add_argument('--batch_size', '-b', type=int, help='batch size')
    # parser.add_argument('--batch_size_valid', default=2, type=int, help='validation mini-batch size')
    parser.add_argument('--epochs', type=int, help='epochs to train')
    parser.add_argument('--optim', default='Adam', help='optimizer to use. only support SGD and Adam')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--lr_per_sample', type=float, help='learning rate per sample')
    parser.add_argument('--lr_pow', default=0.9, type=float, help='power in poly to drop LR')
    parser.add_argument('--warmup_lr', default=1e-6, type=float, help='starting learning rate for warm up')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='numbers of warmup epochs')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=0, type=float, help='factor for L2 regularization')

    parser.add_argument('--valid_visu', default=1, type=int, help='how many batches to be visualized when eval')
    # Misc arguments
    parser.add_argument('--no_cuda', action='store_true', help='disable cuda')
    parser.add_argument('--seed', default=594277, type=int, help='manual seed')
    parser.add_argument('--disp_iter', type=int, default=1, help='iterations frequency to display')
    parser.add_argument('--save_every', type=int, default=25, help='epochs frequency to save state_dict')
    parser.add_argument('--no_multigpus', action='store_true', help='disable data parallel')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    if args.batch_size is not None:
        cfg.OPTIM.BATCH_SIZE = args.batch_size
    if args.lr is not None or args.lr_per_sample is not None:
        if args.lr is not None and args.lr_per_sample is not None:
            assert False, "不能同时指定--lr和--lr_per_sample！"
        if args.lr is not None:
            cfg.OPTIM.LR = args.lr
        if args.lr_per_sample is not None:
            cfg.OPTIM.LR = args.lr_per_sample * cfg.OPTIM.BATCH_SIZE
    if args.epochs is not None:
        cfg.OPTIM.MAX_EPOCH = args.epochs
    if args.num_workers is None:
        args.num_workers = min(max(8, cfg.OPTIM.BATCH_SIZE), os.cpu_count()) if not sys.gettrace() else 0

    device = torch.device('cpu' if args.no_cuda else 'cuda')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.join(args.ckpt, args.id), exist_ok=True)

    # Create dataloader
    dataset_train = PerspectiveDataset(cfg, "train", train_mode=True)
    dataset_train_size = len(dataset_train)
    print("num_workers: " + str(args.num_workers))
    print("batch_size: " + str(cfg.OPTIM.BATCH_SIZE))
    print("train_set_size: " + str(dataset_train_size))
    loader_train = DataLoader(
        dataset_train,
        cfg.OPTIM.BATCH_SIZE,
        collate_fn=dataset_train.collate,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=not args.no_cuda,
        worker_init_fn=worker_init_fn)
    if args.valid_root_dir:
        dataset_valid = PerspectiveDataset(cfg, "valid")
        loader_valid = DataLoader(dataset_valid,
                                  min(cfg.OPTIM.BATCH_SIZE, 4),
                                  collate_fn=dataset_valid.collate,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=args.num_workers,
                                  pin_memory=not args.no_cuda,
                                  worker_init_fn=worker_init_fn)

    # Create model
    if args.pth is not None:
        print('Finetune model is given.')
        print('Ignore --backbone and --no_rnn')
        net = load_trained_model(DMHNet, args.pth, cfg, cfg.MODEL.get("BACKBONE", {}).get("NAME", "drn38"),
                                 not args.no_rnn).to(device)
    else:
        net = DMHNet(cfg, cfg.MODEL.get("BACKBONE", {}).get("NAME", "drn38"), not args.no_rnn).to(device)

    if not args.no_multigpus:
        net = nn.DataParallel(net)  # multi-GPU

    # Create optimizer
    print("LR {:f}".format(cfg.OPTIM.LR))
    if cfg.OPTIM.TYPE == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                              lr=cfg.OPTIM.LR,
                              momentum=args.beta1,
                              weight_decay=args.weight_decay)
    elif cfg.OPTIM.TYPE == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                               lr=cfg.OPTIM.LR,
                               betas=(args.beta1, 0.999),
                               weight_decay=args.weight_decay)
    else:
        raise NotImplementedError()

    # Create tensorboard for monitoring training
    tb_path = os.path.join(args.logs, args.id)
    os.makedirs(tb_path, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_path)

    # Init variable
    args.warmup_iters = args.warmup_epochs * len(loader_train)
    # args.max_iters = args.epochs * len(loader_train)
    # args.running_lr = args.warmup_lr if args.warmup_epochs > 0 else args.lr

    milestones = cfg.OPTIM.get("SCHEDULER", {}).get("MILESTONES", [50, 100])
    gamma = cfg.OPTIM.get("SCHEDULER", {}).get("GAMMA", 0.3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    tb_writer.add_text("cfg", str(cfg))
    tb_writer.add_text("args", str(args))
    tb_writer.add_text("gpuid", os.environ.get("CUDA_VISIBLE_DEVICES", "None"))

    # Init bin mask
    # anglex = np.linspace(-256, 255, 512)
    # angley = np.linspace(256, -255, 512)
    # xv, yv = np.meshgrid(anglex, angley)
    # # idx is the mapping table
    # idx = (np.rad2deg(np.arctan2(xv, yv)) + 180 - 1).astype(int)
    # binary_mask = np.zeros((512, 512, 360))
    # for i in range(360):
    #     binary_mask[np.where(idx == i)[0], np.where(idx == i)[1], i] = 1
    # binary_mask = torch.tensor(binary_mask, dtype=torch.float32)

    best_valid_score = 0  # 筛选最佳模型：以3DIoU为准
    # Start training
    for ith_epoch in trange(1, cfg.OPTIM.MAX_EPOCH + 1, desc='Epoch', unit='ep'):
        # Train phase
        net.train()
        # torch.cuda.empty_cache()
        iterator_train = iter(loader_train)
        cur_sample_count = 0
        for _ in trange(len(loader_train), desc='Train ep%s' % ith_epoch, position=1):
            # Set learning rate
            # adjust_learning_rate(optimizer, args)

            input = next(iterator_train)
            for k in input:
                if isinstance(input[k], torch.Tensor):
                    input[k] = input[k].to(device)

            cur_sample_count += len(input["p_imgs"])
            tb_total_sample_count = (ith_epoch - 1) * dataset_train_size + cur_sample_count

            losses, results_dict = net(input)

            for k, v in losses.items():
                k = 'train/%s' % k
                tb_writer.add_scalar(k, v.item(), tb_total_sample_count)
            loss = losses['total']

            # backprop
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0, norm_type=2)
            optimizer.step()

        tb_writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], ith_epoch)

        # Valid phase
        valid_loss, imgs, metrics = valid(cfg, net, loader_valid, dataset_valid, device, args.valid_visu, valid_epoch=ith_epoch)

        if cfg.get("TEST_METRIC", False):
            dataset_test = PerspectiveDataset(cfg, "test")
            loader_test = DataLoader(dataset_test,
                                     min(cfg.OPTIM.BATCH_SIZE, 4),
                                     collate_fn=dataset_test.collate,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=args.num_workers,
                                     pin_memory=not args.no_cuda,
                                     worker_init_fn=worker_init_fn)
            test_loss, test_imgs, test_metrics = valid(cfg, net, loader_test, dataset_test, device, 0)
            for k, v in test_metrics.items():
                print("{:s} {:f}".format(k, v))
                tb_writer.add_scalar('testmetric/%s' % k, v, ith_epoch)

        for k, v in imgs.items():
            tb_writer.add_image('valid/{:s}'.format(k), v, ith_epoch, dataformats="HWC")

        for k, v in valid_loss.items():
            print("{:s} {:f}".format(k, v))
            tb_writer.add_scalar('valid/%s' % k, v, ith_epoch)

        for k, v in metrics.items():
            print("{:s} {:f}".format(k, v))
            tb_writer.add_scalar('metric/%s' % k, v, ith_epoch)

        # Save best validation loss model
        if "3DIoU" in metrics:
            valid_score = metrics["3DIoU"]
        else:
            valid_score = 100 - valid_loss["total"] # 无后处理训练时，筛选模型使用

        if valid_score >= best_valid_score:
            best_valid_score = valid_score
            print("save BEST VALID ckpt " + str(ith_epoch))
            save_model(net, os.path.join(args.ckpt, args.id, 'best_valid.pth'), args)

        # Periodically save model
        if ith_epoch % args.save_every == 0:
            print("save ckpt " + str(ith_epoch))
            save_model(net, os.path.join(args.ckpt, args.id, 'epoch_%d.pth' % ith_epoch), args)
        scheduler.step()

    if cfg.get("FINAL_EVAL", False):
        print("现在开始finalEval！")
        commandLine = "python eval.py --cfg_file {:s} --ckpt ckpt/{:s}/best_valid.pth --print_detail --output_file".format(args.cfg_file, args.id)
        if cfg.get("FINAL_EVAL_METHOD"):
            commandLine += " --set POST_PROCESS.METHOD {:s}".format(cfg.FINAL_EVAL_METHOD)
        print("要执行的命令行 " + commandLine)
        os.system(commandLine)
