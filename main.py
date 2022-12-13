import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import numpy as np
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
# sys.path.append('/public/home/hpc204711099/yxh/gitrepo/scmoco')

import moco.builder
from moco.builder import ClaireNet
from moco.base_encoder import Encoder, EncoderL2
from moco.config import Config
from moco.dataset import ClaireDataset, print_AnchorInfo
from moco.preprocessing import embPipe

from moco.trainval import load_ckpt, evaluate, train, save_checkpoint, get_args, adjust_learning_rate

import matplotlib.pyplot as plt

configs = Config()

args = get_args()
print(args)
print_freq = args.print_freq

# set training environ
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# set logging dir
sane_ps = ''   
sane_ps += f'knn={args.knn}_alpha={args.alpha}_augSet={args.augment_set}_'   # for construction strategy
sane_ps += f'anc-schedl={args.anchor_schedule}_filter={args.fltr}_yita={args.yita}_'   # for refinment strategy
sane_ps += f'eps={args.epochs}_lr={args.lr}_batch-size={args.batch_size}_'   # training params
sane_ps += f'adjustLr={args.adjustLr}_schedule={args.schedule}'             # training params

log_dir = join(configs.out_root, f'{args.dname}/{sane_ps}')
os.makedirs(log_dir, exist_ok=True)

for ri in range(args.n_repeat):
    os.makedirs(join(log_dir, f'weights{ri+1}'), exist_ok=True)     # folder used to save model weights
    os.makedirs(join(log_dir, f'results{ri+1}'), exist_ok=True)     # folder to save lossCurve and umap results

    # Data loading code
    traindir = os.path.join(configs.data_root, args.dname)
    train_dataset = ClaireDataset(
        traindir,
        mode='train',
        select_hvg=args.select_hvg,
        scale=False,
        knn=args.knn,
        alpha=args.alpha,
        augment_set=args.augment_set,
        exclude_fn=(args.dname!='MouseCellAtlas'),
        verbose=1
        )
    val_dataset = ClaireDataset(
        traindir,
        mode='val',
        select_hvg=args.select_hvg,
        scale=False,
        knn=args.knn,
        alpha=args.alpha,
        verbose=0
        )

    assert np.all(train_dataset.gname == val_dataset.gname), 'unmatched gene names'
    assert np.all(train_dataset.cname == val_dataset.cname), 'unmatched cell names'
    val_dataset.X = train_dataset.X  # eliminate randomness in preprocessing steps
    train_steps = train_dataset.n_sample // args.batch_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.workers, 
        shuffle=True, 
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            num_workers=args.workers, 
            shuffle=False, 
            drop_last=False)

    model = ClaireNet(
        base_encoder=EncoderL2,
        in_dim=train_dataset.n_feature,
        lat_dim=args.lat_dim, 
        block_level=args.block_level,
        args=args
    ) 
    
    model = model.cuda()
    
    if args.optim=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    # momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # start training
    if not args.skip_training:
        print('==========>Start training<==========')
        loss = []
        for epoch in range(args.start_epoch, args.epochs):
            if args.adjustLr:
                adjust_learning_rate(optimizer, epoch, args)

            if epoch in args.anchor_schedule:
                print(f'================== Anchor schedule {epoch}')
                lat_emb, latl2_emb = evaluate(val_loader, model, args)

                print('filtering anchors')
                train_dataset.filter_anchors(latl2_emb,  
                                            fltr=args.fltr,
                                            yita=args.yita
                                            )


                train_dataset.exclude_sampleWithoutMNN(True)
                train_dataset.getMnnDict()

                if train_dataset.type_label is not None:
                    print_AnchorInfo(train_dataset.pairs, train_dataset.batch_label, train_dataset.type_label)

                train_loader = torch.utils.data.DataLoader(
                                                        train_dataset, 
                                                        batch_size=args.batch_size, 
                                                        num_workers=args.workers, 
                                                        shuffle=True, 
                                                        drop_last=True)

            # train one epoch
            lossi = train(train_loader, model, optimizer, epoch, args)
            loss.append(lossi)

            if (epoch+1) % args.save_freq == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    # 'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename=join(log_dir, 'weights{}/checkpoint_{:04d}.pth.tar'.format(ri+1, epoch+1)))

        # fig = plt.plot(loss, label='loss')
        np.save(join(log_dir, f'results{ri+1}/loss.npy'), loss)


    # inference with specified checkpoints
    tmp_emb = {}
    for idx in args.visualize_ckpts:
        model = load_ckpt(model, join(log_dir, f'weights{ri+1}'), idx)
        lat_emb, latl2_emb = evaluate(val_loader, model, args)
        
        ad_lat = embPipe(latl2_emb, train_dataset.metadata) 
        ad_lat.write(join(log_dir, f'results{ri+1}/ad_{idx}.h5ad'))

        tmp_emb[idx] = ad_lat 

    # saving plot
    fig2, axes = plt.subplots(len(args.visualize_ckpts), 2, figsize=(16, 6*len(args.visualize_ckpts)))
    for i, idx in enumerate(args.visualize_ckpts):
        print(f'=====================> {idx}')
        
        sc.pl.umap(tmp_emb[idx], color=[configs.batch_key], show=False, ax=axes[i, 0])

        label_key = configs.label_key if configs.label_key in tmp_emb[idx].columns else configs.batch_key
        sc.pl.umap(tmp_emb[idx], color=[label_key], show=False, ax=axes[i, 1])
        
        axes[i, 0].set_title(f'epoch={idx}')
        axes[i, 1].set_title(f'epoch={idx}')

    fig2.savefig(join(log_dir, f'results{ri+1}/umap.png'), facecolor='white')


