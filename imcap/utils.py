import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# https://github.com/fastai/fastai2/blob/8d798c881c1eda564bdf92079bdfe43b43525767/fastai2/callback/training.py
bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

def set_bn_eval(m:nn.Module):
    "Set bn layers in eval mode for all recursive children of `m`."
    for l in m.children():
        if isinstance(l, bn_types) and not next(l.parameters()).requires_grad:
            l.eval()
        set_bn_eval(l)

def freeze_weights(m):
    for param in m.parameters():
        param.requires_grad_(False)


def subsequent_mask(sz, device):
    mask = torch.ones((sz,sz), device=device, dtype=bool)
    mask.triu_(1)
    return mask

def padding_mask(tgt, pad_idx):
    return (tgt==pad_idx).transpose(0,1)        # transpose for seq_len first, batch


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = self.val = 0

    def update(val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count