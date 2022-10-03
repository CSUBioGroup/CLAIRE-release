import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import sys
import numpy as np
import scanpy as sc

import moco.builder
from moco.config import Config
from moco.dataset import ClaireDataset, print_AnchorInfo
from moco.preprocessing import embPipe
from moco.kbet import calculate_kbet
from moco.evaluation_scib_oldVersion import scib_process  # Old=> scIB, New=>scib

import matplotlib.pyplot as plt
from os.path import join

configs = Config()

def calc_four_metrics(ad):
    ad.obsm['X_emb'] = ad.X
    tmp_res = scib_process(ad,
                            adata_raw=None,
                            batch_key=configs.batch_key,
                            label_key=configs.label_key,
                            silhouette_=True,
                            )
    # calc kbet
    kt = calculate_kbet(ad,
            use_rep='X',
            batch_col=configs.batch_key,
            n_neighbors=15,         # ensure the n larger than the one used before
            calc_knn=True,
            n_jobs=10,
        )[2]
    tmp_res.loc['kBET'] = kt  # 'loc' attention,
    return tmp_res


if __name__=='__main__':
    # dname = 'PBMC'
    dname = sys.argv[1]  # MouseCellAtlas, PBMC, Pancreas, ImmHuman, Lung, Muris
    out_dir = join(configs.out_root, dname)
    ckpt_fds = os.listdir(out_dir)

    for fdi in ckpt_fds:
        print('reading ', fdi)
        eval_fd = join(out_dir, fdi, 'eval_folder')
        os.makedirs(eval_fd, exist_ok=True)

        for i in range(1, 6):
            _dir = join(out_dir, fdi, f'results{i}')
            print(f'\t results{i}')

            if not os.path.exists(_dir):
                continue

            ads = list(filter(lambda x: x.startswith('ad'), os.listdir(_dir)))
            ads = sorted(ads)

            scib_res, ckpts = None, []
            for adi in ads:
                ad_path = join(_dir, adi)
                adx = sc.read_h5ad(ad_path)
                tmp_res = calc_four_metrics(adx)

                # merge 
                scib_res = tmp_res if scib_res is None else scib_res.merge(tmp_res, left_index=True, right_index=True, how='outer')
                ckpts.append(adi[:-5].split('_')[1])

            scib_res.columns = ckpts
            scib_res.to_csv(f'{eval_fd}/eval{i}.csv', index=True)



