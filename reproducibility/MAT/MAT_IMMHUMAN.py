import os
import sys
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
from MAT2 import *

import sys
from pathlib import Path
from os.path import join
cur_dir = Path(os.getcwd())
sys.path.append(str(cur_dir.parent.parent.absolute()))

from moco.kbet import calculate_kbet
from moco.utils import py_read_data, load_meta_txt
from moco.evaluation import scib_process 
from moco.prepare_dataset import prepare_dataset
from moco.preprocessing import preprocess_dataset, hvgPipe

from scib_eval import scib_eval

data_dir = '/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Script/sapling/GLOBE/data'
out_dir = '/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Output'

# ================================
# prepare datasets
# ================================
dno = 'ImmHuman'
# sc.settings.set_figure_params(dpi=80)
# sc.settings.figdir = Path(f'/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Output/dataset4/MAT')
# sc.settings.figdir.mkdir(parents=True, exist_ok=True)

n_hvgs = 2000
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


adata = sc.AnnData(X)
adata.var_names = gene_name
adata.obs = df_meta.copy()

data = pd.DataFrame(adata.X.A.T, index=adata.var_names, columns=adata.obs_names)

# prepare anchors
anchor = pd.read_csv(join(dataset_dir, f'seuratAnchors.csv'), header=0, index_col=0)
name2idx = dict(zip(adata.obs_names, np.arange(adata.shape[0])))
anchor['cell1'] = anchor.name1.apply(lambda x:name2idx[x])
anchor['cell2'] = anchor.name2.apply(lambda x:name2idx[x])


# ================================
# training Unsupervised model
# ================================
print('========Unsupervised training========')
lr = 1e-3
eps = [0, 10, 20, 40, 80, 100, 120]
EMBS = []

log_dir = f'./outputs/{dno}-lr={lr}'
n_repeat = 3

for ri in range(n_repeat):
    os.makedirs(join(log_dir, f'results{ri+1}'), exist_ok=True)

    for ep in eps:
        print('training params ', lr, ep)
        model_unv = BuildMAT2(
                            data=data,
                            metadata=df_meta,
                            anchor=anchor,
                            num_workers=6,
                            use_gpu=True,
                            mode='manual',
                            latent_num = 20,
                            learning_rate = lr,
                            batch_size = 256,
                            norm = 'l1',
                            weight_decay = 0.01)
        enc_loss, dec_loss = model_unv.train(epochs=ep, curve=True, dec_train=True)

        rec_unv = model_unv.evaluate(data)
        EMBS.append(rec_unv)

    np.save(join(log_dir, f'results{ri+1}', 'enc_loss.npy'), enc_loss)
    np.save(join(log_dir, f'results{ri+1}', 'dec_loss.npy'), dec_loss)

    # ================================
    # evaluation
    # ================================
    RES = None
    ADAS = []
    for i,ep in enumerate(eps):
        ad_tmp = hvgPipe(EMBS[i], meta=df_meta, scale=True, n_neighbors=15, npcs=50, umap=True)
        ad_tmp.obsm['X_emb'] = ad_tmp.obsm['X_pca']
        ad_tmp.write(join(log_dir, f'results{ri+1}/ad_{ep}.h5ad'))
        ADAS.append(ad_tmp)

        tmp_res = scib_eval(ad_tmp, batch_key, label_key)

        RES = tmp_res if RES is None else RES.merge(tmp_res, left_index=True, right_index=True, how='inner')

    RES.columns = eps
    RES.to_csv(join(log_dir, f'results{ri+1}/eval.csv'), index=True)

    # ================================
    # Show plots
    # ================================
    fig2, axes = plt.subplots(len(eps), 2, figsize=(16, 6*len(eps)))
    for i, idx in enumerate(eps):
        print(f'=====================> {idx}')

        sc.pl.umap(ADAS[i], color=[batch_key], show=False, ax=axes[i, 0])
        sc.pl.umap(ADAS[i], color=[label_key], show=False, ax=axes[i, 1])

        axes[i, 0].set_title(f'epoch={idx}')
        axes[i, 1].set_title(f'epoch={idx}')

    fig2.savefig(join(log_dir, f'results{ri+1}/umap.png'), facecolor='white')