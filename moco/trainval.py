import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import scanpy as sc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import sys
from pathlib import Path
from os.path import join
# cur_dir = Path(os.getcwd())
# sys.path.append(str(cur_dir.parent.absolute()))

import moco.builder
from moco.builder import ClaireNet
from moco.base_encoder import EncoderL2
from moco.config import Config
from moco.dataset import ClaireDataset
from moco.preprocessing import embPipe

configs = Config()

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch scMoCo Training')
    # environ settings
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id to use.')
    parser.add_argument('--workers', default=10, type=int,
                        help='number of worker for dataloader')
    parser.add_argument('--seed', default=39, type=int,
                        help='random seed')

    # repeat 
    parser.add_argument('--n_repeat', default=1, type=int,
                        help='repeat times')

    # model archs
    parser.add_argument('--lat_dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--block_level', default=1, type=int,
                        help='feature dimension (default: 1)')
    parser.add_argument('--moco_k', default=2048, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco_t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--init', default='uniform', type=str,
                        help='weight init')

    # dataset
    parser.add_argument('--dname', default='Pancreas', type=str,
                        help='dataset name')
    parser.add_argument('--select_hvg', default=2000, type=int, 
                        help='number of hvgs')                  # 5000 for Muris dataset
    parser.add_argument('--scale', action='store_true', 
                        help='whether to scale data in preprocessing')
    parser.add_argument('--knn', default=10, type=int, 
                        help='knn for positive augmentation')
    parser.add_argument('--alpha', default=.99, type=float,
                        help='mixup left threshold')
    parser.add_argument('--augment_set', default=[], nargs='*', type=str,
                        help='augmentation operation')

    parser.add_argument('--anchor_schedule', default=[], nargs='*', type=int,
                        help='')
    parser.add_argument('--fltr', default='gmm', type=str,
                        help='which filter to apply, gmm or naive')
    parser.add_argument('--yita', default=0.5, type=float,
                        help='filtering params')

    # training params
    parser.add_argument('--skip_training', action='store_true',
                        help='initial learning rate')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--adjustLr', action='store_true', 
                        help='whether adjust learning rate during learning')
    parser.add_argument('--cos', action='store_true', 
                        help='cosine learning rate schedule')
    parser.add_argument('--schedule', default=[10], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--optim', default='Adam', type=str,
                        help='optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='momentum of SGD solver')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='weight decay (default: 1e-4)')

    # loss
    parser.add_argument('--symmetric', default=True, type=bool,
                        help='symmetric loss')

    parser.add_argument('--epochs', default=80, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')

    # logging settings
    parser.add_argument('--visualize_ckpts', default=[10, 20, 40, 80, 120], nargs='*', type=int,
                        help='select ckpts to visualize their embeddings')
    parser.add_argument('--print_freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--save_freq', default=10, type=int,
                        help='print frequency (default: 10)')


    args = parser.parse_args()
    return args


def load_ckpt(model, log_dir, ckpt_idx):
    # print("=> loading checkpoint '{}'".format(args.resume))
    ckpt_name = 'checkpoint_{:04d}.pth.tar'.format(ckpt_idx)
    checkpoint = torch.load(join(log_dir, ckpt_name))

    model.load_state_dict(checkpoint['state_dict'])

    print("=> loaded checkpoint: {}"
          .format(ckpt_name))

    return model


def train(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (cells, indices) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        cells_q, cells_k = cells[0].cuda(), cells[1].cuda()

        # compute output
        loss = model(cells_q, cells_k)

        losses.update(loss.item(), cells_q.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)

    return losses.avg


def evaluate(val_loader, model, args):
    model.eval()

    lat_emb, latl2_emb = torch.Tensor(0).cuda(), torch.Tensor(0).cuda()
    for i, (cells, indices) in enumerate(val_loader):
        # measure data loading time
        cells_q = cells[0].cuda()

        # compute embedding from lat OR from out 
        embeddings1 = model.encoder_q.encoder(cells_q)  # from lat
        embeddings2 = model.encoder_q(cells_q)        # lat + L2

        lat_emb = torch.cat([lat_emb, embeddings1], 0)
        latl2_emb = torch.cat([latl2_emb, embeddings2], 0)

    lat_emb = lat_emb.detach().cpu().numpy()
    latl2_emb = latl2_emb.detach().cpu().numpy()

    return lat_emb, latl2_emb


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr