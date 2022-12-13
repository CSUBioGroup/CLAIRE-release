import os
import sys
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import scipy.sparse as sps

from os.path import join
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import sys
from pathlib import Path
from os.path import join
cur_dir = Path(os.getcwd())
sys.path.append(str(cur_dir.parent.parent.absolute()))
from tnn.tnn import *

from moco.kbet import calculate_kbet
from moco.utils import py_read_data, load_meta_txt
from moco.evaluation import scib_process 
from moco.prepare_dataset import prepare_dataset
from moco.preprocessing import preprocess_dataset
from scib_eval import scib_eval

from memory_profiler import profile

data_dir = '/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Script/sapling/GLOBE/data'
out_dir = '/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Output'


def main_workers(n_cells=2000):
    

    # ================================
    # prepare datasets
    # ================================
    dno = f'Muris_{n_cells}'

    n_hvgs = 5000
    scale = False
    batch_key = 'batchlb'
    label_key = 'CellType'

    dataset_dir = join(data_dir, dno)
    sps_x, genes, cells, df_meta = prepare_dataset(dataset_dir)
    X, cell_name, gene_name, df_meta = preprocess_dataset(
        sps_x, 
        cells, 
        genes, 
        df_meta, 
        n_hvgs, 
        scale, 
    )
    
    st = datetime.datetime.now()
    adata = sc.AnnData(X)
    adata.var_names = gene_name
    adata.obs = df_meta.copy()

    sc.pp.scale(adata, max_value=None)
    sc.tl.pca(adata, n_comps=50)

    # repeat 
    lr = 1e-3
    log_dir = f'./outputs/{dno}-lr={lr}'
    n_repeat = 3

    # training
    EPS = [0, 10, 20, 40, 80, 100, 120]
    EMBS_INSCT = []
    RES_INSCT = None

    ad_ep = adata
    
    ep = 80
    model = TNN(lr=lr, epochs=ep, k=150, n_epochs_without_progress=ep)  # default lr=1e-3
    model.fit(X = ad_ep, batch_name=batch_key)

    tmp_emb = model.transform(X = ad_ep)
    ed = datetime.datetime.now()
    
    print('=========================')
    print('N={}, time_cost={:.4f}'.format(n_cells, (ed-st).total_seconds() * 1.))
    print('=========================')

@profile
def main():
    main_workers(n_cells=120000)
    
if __name__ == "__main__":
    main()
