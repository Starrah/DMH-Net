import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def group_weight(module):
    # Group module parameters into two group
    # One need weight_decay and the other doesn't
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(
        module.parameters())) == len(group_decay) + len(group_no_decay)
    return [
        dict(params=group_decay),
        dict(params=group_no_decay, weight_decay=.0)
    ]


def adjust_learning_rate(optimizer, args):
    if args.cur_iter < args.warmup_iters:
        frac = args.cur_iter / args.warmup_iters
        step = args.lr - args.warmup_lr
        args.running_lr = args.warmup_lr + step * frac
    else:
        frac = (float(args.cur_iter) - args.warmup_iters) / (
            args.max_iters - args.warmup_iters)
        scale_running_lr = max((1. - frac), 0.)**args.lr_pow
        args.running_lr = args.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.running_lr


def save_model(net, path, args):
    state_dict = OrderedDict({
        'args': args.__dict__,
        'kwargs': {
            'backbone': net.module.backbone,
            'use_rnn': net.module.use_rnn,
        },
        'state_dict': net.state_dict(),
    })
    pipesave(state_dict, path)


def load_trained_model(Net, path, *args):
    state_dict = pipeload(path, map_location='cpu')
    net = Net(*args)
    net.load_state_dict(state_dict['state_dict'], strict=False)
    return net


def pipeload(filepath: str, **kwargs):
    if not filepath.startswith("hdfs://"):
        return torch.load(filepath, **kwargs)
    with hopen(filepath, "rb") as reader:
        accessor = io.BytesIO(reader.read())
        state_dict = torch.load(accessor, **kwargs)
        del accessor
        return state_dict

def pipesave(obj, filepath: str, **kwargs):
    if filepath.startswith("hdfs://"):
        with hopen(filepath, "wb") as writer:
            torch.save(obj, writer, **kwargs)
    else:
        torch.save(obj, filepath, **kwargs)


HADOOP_BIN = 'PATH=/usr/bin:$PATH hdfs'
from contextlib import contextmanager
@contextmanager
def hopen(hdfs_path, mode="r"):
    pipe = None
    if mode.startswith("r"):
        pipe = subprocess.Popen(
            "{} dfs -text {}".format(HADOOP_BIN, hdfs_path), shell=True, stdout=subprocess.PIPE)
        yield pipe.stdout
        pipe.stdout.close()
        pipe.wait()
        return
    if mode == "wa":
        pipe = subprocess.Popen(
            "{} dfs -appendToFile - {}".format(HADOOP_BIN, hdfs_path), shell=True, stdin=subprocess.PIPE)
        yield pipe.stdin 
        pipe.stdin.close()
        pipe.wait()
        return
    if mode.startswith("w"):
        pipe = subprocess.Popen(
            "{} dfs -put -f - {}".format(HADOOP_BIN, hdfs_path), shell=True, stdin=subprocess.PIPE)
        yield pipe.stdin
        pipe.stdin.close()
        pipe.wait()
        return
    raise RuntimeError("unsupported io mode: {}".format(mode))

def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    if get_softmax:
        p_output = F.softmax(p_output, -1)
        q_output = F.softmax(q_output, -1)
    log_mean_output = ((p_output + q_output )/2).log()
    ploss = F.kl_div(log_mean_output, p_output, reduction='batchmean')
    qloss = F.kl_div(log_mean_output, q_output, reduction='batchmean')
    return (ploss + qloss) / 2