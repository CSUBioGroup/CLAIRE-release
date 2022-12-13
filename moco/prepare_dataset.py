import torch
from torch.utils.data import Dataset

import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps

from os.path import join

from moco.utils import py_read_data, load_meta_txt, load_meta_txt7
from moco.config import Config

configs = Config()


def prepare_MouseCellAtlas(data_root):
    data_name = 'filtered_total_batch1_seqwell_batch2_10x'

    sps_x, gene_name, cell_name = py_read_data(data_root, data_name)
    df_meta = load_meta_txt(join(data_root, 'filtered_total_sample_ext_organ_celltype_batch.txt'))
    df_meta['CellType'] = df_meta['ct']

    df_meta[configs.batch_key] = df_meta[configs.batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[configs.label_key].astype('category')

    return sps_x, gene_name, cell_name, df_meta

def prepare_Pancreas(data_root):
    data_name = 'myData_pancreatic_5batches'

    sps_x, gene_name, cell_name = py_read_data(data_root, data_name)
    df_meta = load_meta_txt(join(data_root, 'mySample_pancreatic_5batches.txt'))
    df_meta['CellType'] = df_meta['celltype']

    df_meta[configs.batch_key] = df_meta[configs.batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[configs.label_key].astype('category')

    return sps_x, gene_name, cell_name, df_meta

def prepare_PBMC(data_root):
    sps_x1, gene_name1, cell_name1 = py_read_data(data_root, 'b1_exprs')
    sps_x2, gene_name2, cell_name2 = py_read_data(data_root, 'b2_exprs')

    sps_x = sps.hstack([sps_x1, sps_x2])
    cell_name = np.hstack((cell_name1, cell_name2))

    assert np.all(gene_name1 == gene_name2), 'gene order not match'
    gene_name = gene_name1

    df_meta1 = load_meta_txt(join(data_root, 'b1_celltype.txt'))
    df_meta2 = load_meta_txt(join(data_root, 'b2_celltype.txt'))
    df_meta1['batchlb'] = 'Batch1'
    df_meta2['batchlb'] = 'Batch2'

    df_meta = pd.concat([df_meta1, df_meta2])

    df_meta[configs.batch_key] = df_meta[configs.batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[configs.label_key].astype('category')

    return sps_x, gene_name, cell_name, df_meta

def prepare_CellLine(data_root):
    b1_exprs_filename = "b1_exprs"
    b2_exprs_filename = "b2_exprs"
    b3_exprs_filename = "b3_exprs"
    b1_celltype_filename = "b1_celltype.txt"
    b2_celltype_filename = "b2_celltype.txt"
    b3_celltype_filename = "b3_celltype.txt"

    # data_name = 'b1_exprs'
    batch_key = 'batchlb'
    label_key = 'CellType'

    expr_mat1, g1, c1 = py_read_data(data_root, b1_exprs_filename)
    metadata1 = pd.read_csv(join(data_root, b1_celltype_filename), sep="\t", index_col=0)

    # expr_mat2 = pd.read_csv(join(data_dir, b2_exprs_filename), sep="\t", index_col=0).T
    expr_mat2, g2, c2 = py_read_data(data_root, b2_exprs_filename)
    metadata2 = pd.read_csv(join(data_root, b2_celltype_filename), sep="\t", index_col=0)

    expr_mat3, g3, c3 = py_read_data(data_root, b3_exprs_filename)
    metadata3 = pd.read_csv(join(data_root, b3_celltype_filename), sep="\t", index_col=0)

    metadata1['batchlb'] = 'Batch_1'
    metadata2['batchlb'] = 'Batch_2'
    metadata3['batchlb'] = 'Batch_3'

    assert np.all(g1 == g2), 'gene name not match'

    cell_name = np.hstack([c1, c2, c3])
    gene_name = g1

    df_meta = pd.concat([metadata1, metadata2, metadata3])
    sps_x = sps.hstack([expr_mat1, expr_mat2, expr_mat3])

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return sps_x, gene_name, cell_name, df_meta 

def prepare_MouseRetina(data_root):
    b1_exprs_filename = "b1_exprs"
    b2_exprs_filename = "b2_exprs"
    b1_celltype_filename = "b1_celltype.txt"
    b2_celltype_filename = "b2_celltype.txt"

    # data_name = 'b1_exprs'
    batch_key = 'batchlb'
    label_key = 'CellType'

    expr_mat1, g1, c1 = py_read_data(data_root, b1_exprs_filename)
    metadata1 = pd.read_csv(join(data_root, b1_celltype_filename), sep="\t", index_col=0)

    # expr_mat2 = pd.read_csv(join(data_root, b2_exprs_filename), sep="\t", index_col=0).T
    expr_mat2, g2, c2 = py_read_data(data_root, b2_exprs_filename)
    metadata2 = pd.read_csv(join(data_root, b2_celltype_filename), sep="\t", index_col=0)

    metadata1['batchlb'] = 'Batch_1'
    metadata2['batchlb'] = 'Batch_2'

    assert np.all(g1 == g2), 'gene name not match'

    cell_name = np.hstack([c1, c2])
    gene_name = g1

    df_meta = pd.concat([metadata1, metadata2])
    sps_x = sps.hstack([expr_mat1, expr_mat2])

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return sps_x, gene_name, cell_name, df_meta 

def prepare_Simulation(data_root):
    # data_root: /home/.../Data/dataset3/simul1_dropout_005_b1_500_b2_900
    batch_key = 'Batch'
    label_key = 'Group'

    # manually switch to counts_all.txt
    # ensure row is gene
    X = pd.read_csv(join(data_root, 'counts.txt'), sep='\t',header=0, index_col=0)  # row is cell
    X = X.T   # to gene

    metadata = pd.read_csv(join(data_root, 'cellinfo.txt'), header=0, index_col=0, sep='\t')
    metadata[configs.batch_key] = metadata[batch_key]
    metadata[configs.label_key] = metadata[label_key]

    return X, X.index.values, X.columns.values, metadata

def prepare_Lung(data_root):
    # data_root: /home/.../Data/dataset3/simul1_dropout_005_b1_500_b2_900
    batch_key = 'batch'
    label_key = 'cell_type'

    # ensure row is gene
    adata = sc.read_h5ad(join(data_root, 'Lung_atlas_public.h5ad'))

    X = adata.layers['counts'].A.T  # gene by cell

    gene_name = adata.var_names
    cell_name = adata.obs_names.values
    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_ImmHuman(data_root):
    batch_key = 'batch'
    label_key = 'final_annotation'
    pseudo_key = 'dpt_pseudotime'

    # ensure row is gene
    adata = sc.read_h5ad(join(data_root, 'Immune_ALL_human.h5ad'))

    X = sps.csr_matrix(adata.layers['counts'].T)  # gene by cell

    gene_name = adata.var_names
    cell_name = adata.obs_names.values
    df_meta = adata.obs[[batch_key, label_key, pseudo_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_ImmHumanMouse(data_root):
    batch_key = 'batch'
    label_key = 'final_annotation'
    pseudo_key = 'dpt_pseudotime'

    # ensure row is gene
    adata = sc.read_h5ad(join(data_root, 'Immune_ALL_hum_mou_filter.h5ad'))

    X = sps.csr_matrix(adata.layers['counts'].T)  # gene by cell

    gene_name = adata.var_names
    cell_name = adata.obs_names.values
    df_meta = adata.obs[[batch_key, label_key, pseudo_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Mixology(data_root):
    batch_key = 'batch'
    label_key = 'cell_line_demuxlet'

    # ensure row is gene
    adata = sc.read_h5ad(join(data_root, 'sc_mixology.h5ad'))
    adata = adata[:, adata.var.Selected.values.astype('bool')].copy()  # use selected hvg, 2000

    # X = sps.csr_matrix(adata.layers['norm_data'].T)  # gene by cell
    X = sps.csr_matrix(adata.X.T)  # gene by cell

    gene_name = adata.var_names
    cell_name = adata.obs_names.values
    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Muris(data_root):
    batch_key = 'batch'
    label_key = 'cell_ontology_class'

    adata = sc.read_h5ad(join(data_root, 'muris_sample_filter.h5ad'))  # 6w cells

    X = sps.csr_matrix(adata.layers['counts'].T)

    gene_name = adata.var_names
    cell_name = adata.obs_names.values

    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Muris_2000(data_root):
    batch_key = 'batch'
    label_key = 'cell_ontology_class'

    adata = sc.read_h5ad(join(data_root, 'muris_sample_2000.h5ad'))  # 6w cells

    X = sps.csr_matrix(adata.layers['counts'].T)

    gene_name = adata.var_names
    cell_name = adata.obs_names.values

    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Muris_4000(data_root):
    batch_key = 'batch'
    label_key = 'cell_ontology_class'

    adata = sc.read_h5ad(join(data_root, 'muris_sample_4000.h5ad'))  # 6w cells

    X = sps.csr_matrix(adata.layers['counts'].T)

    gene_name = adata.var_names
    cell_name = adata.obs_names.values

    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Muris_8000(data_root):
    batch_key = 'batch'
    label_key = 'cell_ontology_class'

    adata = sc.read_h5ad(join(data_root, 'muris_sample_8000.h5ad'))  # 6w cells

    X = sps.csr_matrix(adata.layers['counts'].T)

    gene_name = adata.var_names
    cell_name = adata.obs_names.values

    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Muris_16000(data_root):
    batch_key = 'batch'
    label_key = 'cell_ontology_class'

    adata = sc.read_h5ad(join(data_root, 'muris_sample_16000.h5ad'))  # 6w cells

    X = sps.csr_matrix(adata.layers['counts'].T)

    gene_name = adata.var_names
    cell_name = adata.obs_names.values

    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Muris_30000(data_root):
    batch_key = 'batch'
    label_key = 'cell_ontology_class'

    adata = sc.read_h5ad(join(data_root, 'muris_sample_30000.h5ad'))  # 6w cells

    X = sps.csr_matrix(adata.layers['counts'].T)

    gene_name = adata.var_names
    cell_name = adata.obs_names.values

    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Muris_60000(data_root):
    batch_key = 'batch'
    label_key = 'cell_ontology_class'

    adata = sc.read_h5ad(join(data_root, 'muris_sample_60000.h5ad'))  # 6w cells

    X = sps.csr_matrix(adata.layers['counts'].T)

    gene_name = adata.var_names
    cell_name = adata.obs_names.values

    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Muris_120000(data_root):
    batch_key = 'batch'
    label_key = 'cell_ontology_class'

    adata = sc.read_h5ad(join(data_root, 'muris_sample_120000.h5ad'))  # 6w cells

    X = sps.csr_matrix(adata.layers['counts'].T)

    gene_name = adata.var_names
    cell_name = adata.obs_names.values

    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Neo(data_root):
    # batch_key = 'batch'
    label_key = 'grouping'

    import h5py
    filename = join(data_root, 'mouse_brain_merged.h5')

    with h5py.File(filename, "r") as f:
        # List all groups
        cell_name = list(map(lambda x:x.decode('utf-8'), f['cell_ids'][...]))
        gene_name = list(map(lambda x:x.decode('utf-8'), f['gene_names'][...]))
        
        X = sps.csr_matrix(f['count'][...].T)  # transpose to (genes, cells)
        types = list(map(lambda x:x.decode('utf-8'), f['grouping'][...]))

    df_meta = pd.DataFrame(types, index=cell_name, columns=[configs.label_key])
    df_meta[configs.batch_key] = 'Batch_B'
    df_meta.iloc[:10261, -1] = 'Batch_A'     
    return X, gene_name, cell_name, df_meta

def prepare_PBMCMultome(data_root):
    label_key = 'seurat_annotations'

    adata_rna = sc.read_h5ad(join(data_root, 'RNA/adata_rna.h5ad'))
    adata_atac = sc.read_h5ad(join(data_root, 'ATAC_GAM/adata_atac_gam.h5ad'))

    share_gene = np.intersect1d(adata_rna.var_names, adata_atac.var_names)
    adata_rna = adata_rna[:, share_gene]
    adata_atac = adata_atac[:, share_gene]

    X = sps.vstack([adata_rna.X, adata_atac.X]).T
    X = sps.csr_matrix(X)

    meta1 = pd.read_csv(join(data_root, 'metadata.csv'), index_col=0)
    meta1[configs.label_key] = meta1[label_key]
    meta1[configs.batch_key] = 'RNA'
    meta2 = meta1.copy()
    meta2[configs.batch_key] = 'ATAC'

    meta1.index = [f'{_}_reference' for _ in meta1.index] 
    meta2.index = [f'{_}_query' for _ in meta2.index]
    meta = pd.concat([meta1, meta2])

    cname = np.array(meta.index)
    return X, share_gene, cname, meta


def prepare_dataset(data_dir):
    dataset_name = data_dir.split('/')[-1]
    func_dict = {
                    'MouseCellAtlas': prepare_MouseCellAtlas, 
                    'Pancreas': prepare_Pancreas, 
                    'PBMC': prepare_PBMC, 
                    'CellLine': prepare_CellLine, 
                    'MouseRetina': prepare_MouseRetina, 
                    'Lung': prepare_Lung,
                    'ImmHuman': prepare_ImmHuman,
                    'Muris': prepare_Muris,
                    'Neocortex': prepare_Neo,
                    'Muris_2000': prepare_Muris_2000,
                    'Muris_4000': prepare_Muris_4000,
                    'Muris_8000': prepare_Muris_8000,
                    'Muris_16000': prepare_Muris_16000,
                    'Muris_30000': prepare_Muris_30000,
                    'Muris_60000': prepare_Muris_60000,
                    'Muris_120000': prepare_Muris_120000,
                    'PBMCMultome': prepare_PBMCMultome,
                    # 'Simuation2': prepare_Simulation2
    }

    # dataset 3 
    return func_dict.get(dataset_name, prepare_Simulation)(data_dir)





