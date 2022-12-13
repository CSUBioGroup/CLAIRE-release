# CLAIRE
contrastive learning-based batch correction framework for better balance between batch mixing and preservation of cellular heterogeneity

## Installation

Ensure Pytorch is installed in your python environment (our test version: pytorch 1.7.1). Then installing the dependencies:
```
pip install -r requirements.txt
```

## Datasets
All datasets used in our paper can be found in [`zenodo`](https://zenodo.org/record/7136754)

Data source:

Pancreas, PBMC, MouseCellAtlas: [`link`](https://github.com/JinmiaoChenLab/Batch-effect-removal-benchmarking/tree/master/Data)

Muris: [`link`](https://drive.google.com/uc?id=17ou8nVfrTYXJhA_a-OJOEm03zfbfBgxH)

ImmHuman, Lung: [`link`](https://github.com/theislab/scib-reproducibility)

Neocortex: [`link`](https://github.com/jaydu1/VITAE/tree/master/data)

PBMCMultome: [`link`](https://support.10xgenomics.com/single-cell-multiome-atac-gex/datasets/1.0.0/pbmc_granulocyte_sorted_10k)

## Tutorial
We provide multiple tutorials for users to apply CLAIRE in different downstream analysis tasks.
* Clustering and finding DEGs [`1`](./demos/Clustering_and_finding_DEGs.ipynb)
* Label transfer between scRNA-seq datasets [`2`](./demos/Label-Transfer-PBMC.ipynb)
* Label transfer across scRNA-seq and scATAC-seq [`3`](./demos/Label-Transfer-CrossOmics.ipynb)
* Trajectory Analysis [`4`](./demos/Trajectory_Analysis.ipynb)

## Usage
Run ```sh train_script.sh``` to reproduce our outputs. 
Then, run ```python eval.py [dataset_name]``` to evaluate the outputs. 

Following is instructions about how to use CLAIRE for new datasets:

### 1. Preparing data
Data folder structure:

```latex
|-- data
|   |-- Pancreas
|   |   |-- data.mtx
|   |   |-- metadata.csv
|   |   |-- gene_names.csv
|   |-- ImmHuman
|   |   |-- adata_data.h5ad
|   |-- PBMC
|   |   |-- data.npz
|   |   |-- metadata.csv
|   |   |-- gene_names.csv
|   |-- new_dataset
|   |   |-- ...
...
```

Assume that the new dataset is saved in the 'new_dataset' folder. Modify `prepare_NewDataset` function in [`prepare_dataset.py`](./moco/prepare_dataset.py#L419)

```Python
def prepare_NewDataset(data_root):  
   '''
      return:
         X:         scipy.sparse.csr_matrix, row = feature, column = cell
         gene_name: array of feature (gene) names
         cell_name: array of cell (barcodes) names
         df_meta:   metadata of dataset, columns include 'batchlb'(batch column), 'CellType'(optional)
   '''
   # =========== 
   # example

   # batch_key = 'batch'
   # label_key = 'final_annotation'

   # adata = sc.read_h5ad(join(data_root, 'Immune_ALL_human.h5ad'))  # read ImmHuman dataset
   # X = sps.csr_matrix(adata.layers['counts'].T)  # gene by cell

   # gene_name = adata.var_names.values
   # cell_name = adata.obs_names.values
   # df_meta = adata.obs[[batch_key, label_key]].copy()

   # df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
   # df_meta[configs.label_key] = df_meta[label_key].astype('category')

   # return X, gene_name, cell_name, df_meta
   # ===========
```

Then, modify the last key of dictionary in [`prepare_dataset`](./moco/prepare_dataset.py#L467) function. Under our assumption, the last key should be 'new_dataset'.

### 2. Compute Mutual Nearest neighbors (MNNs) between batches
We provide two approaches to obtain MNNs. 
1. (Recommended) Follow the pipeline in [`find_anchors.R`](./moco/find_anchors.R)

   a) read data into `expr_mat` and `metadata`

   b) input `expr_mat` and `metadat` into `find_anchors` function

   c) MNNs will be exported as a csv file and place the exported file in the 'new_dataset' folder.

2. Compute anchors in python [`computeAnchors`](./moco/sNNs.py#L317) (CCA + MNN)

   a) Uncomment line 80 in [`dataset.py`](./moco/dataset.py#L80) or uncomment line 80 in [`dataset_optim.py`](./moco/dataset_optim.py#L80)
   
   b) comment out line 83 in [`dataset.py`](./moco/dataset.py#L83) or comment out line 83 in [`dataset_optim.py`](./moco/dataset_optim.py#L83)

### 3. Run CLAIRE
run the following command in terminal:
``` Python
python main.py --dname 'new_dataset' 
            --n_repeat 3
            --moco_k 2048 --moco_m 0.999 --moco_t 0.07 \
            --block_level 1 --lat_dim 128 --symmetric True \
            --select_hvg 2000 \
            --knn 10 --alpha 0.5 --augment_set 'int' \
            --anchor_schedule 4 --fltr 'gmm' --yita 0.5 \
            --lr 1e-4 --optim Adam --weight_decay 1e-5 --epochs 120 --batch_size 256\
            --workers 6 --init 'uniform' \
            --visualize_ckpts 10 20 40 80 120
```

or 

``` Python
python main_optim.py --dname 'new_dataset' 
            --n_repeat 3
            --moco_k 2048 --moco_m 0.999 --moco_t 0.07 \
            --block_level 1 --lat_dim 128 --symmetric True \
            --select_hvg 2000 \
            --knn 10 --alpha 0.5 --augment_set 'int' \
            --anchor_schedule 4 --fltr 'gmm' --yita 0.5 \
            --lr 1e-4 --optim Adam --weight_decay 1e-5 --epochs 120 --batch_size 256\
            --workers 6 --init 'uniform' \
            --visualize_ckpts 10 20 40 80 120
```

main_optim.py and main.py produce similar results but main_optim.py runs twice as fast as main.py. 

### 4. Output
The output will be saved in `outputs/new_dataset` folder. Output structure:
```latex
|-- outputs
|   |-- new_dataset
|   |   |-- parameter_setting
|   |   |   |-- results1
|   |   |   |   |-- ad_0.h5ad
|   |   |   |   |-- ad_10.h5ad
|   |   |   |   |-- ad_20.h5ad
|   |   |   |   |-- ...
|   |   |   |   |-- loss.npy
|   |   |   |   |-- umap.png
|   |   |   |-- weights1
|   |   |   |   |-- checkpoint_0010.pth.tar
|   |   |   |   |-- checkpoint_0020.pth.tar
|   |   |   |   |-- ...
|   |   |   |-- results2
|   |   |   |-- weights2
...
```

where weights1 saves trained model weights and result1 saves integrated low-dimensional embeddings in a AnnData object. Results2 and weights2 indicate repeated experiment.

## Arguments
The training command contains a number of parameters, most of them can be fixed, some of them should be modified according to the dataset.
* `dname`: data folder name 
* `n_repeat`: number of repeated experiments
* `moco_k`, `moco_m`, `moco_t`, `block_level`, `lat_dim`, `symmetric`: Moco architecture parameters. Fixed
* `select_hvg`: number of highly variable genes to select, use None to keep all genes.
* `knn`: number of neighbors for postive mixup, Fixed
* `alpha`: parameter of uniform distribution, U(alpha, 1). Fixed
* `augment_set`: which operation to use for mixup. Fixed
* `anchor_schedule`: which epoch to filter anchors, preferably between [2,10]
* `fltr`: use gaussian mixture model to filter anchors. Fixed
* `yita`: threshold for filtering anchors, preferably between [0.4, 0.6]
* `lr`： learning rate for model training, preferably 1e-4
* `optim`: optimizer. Fixed
* `weight_decay`: Fixed
* `epochs`: number of training epochs, preferably between [20, 120]
* `batch_size`: training and test batch size, preferably in {256, 512}
* `workers`: number of workers for dataloader. Fixed
* `init`: model initialization strategy. Fixed
* `visualize_ckpts`: which epochs to visualize embeddings.

We highly recommend users to inspect the umap visualizations along training epochs. When batches are mixed sufficiently, training should be stopped immediately. 
If batches are not sufficently mixed, we recommend: 
* `yita` parameter can be set smaller to allow more MNN retained
* `anchor_schedule` can be set higher
* `epochs` can be set larger

If cellular heterogeneity is ambiguous, we recommend:
* `yita` parameter can be set larger to retain more precise MNNs.
* `anchor_schedule` can be set lower
* `epochs` can be set smaller


## Note

There are several versions of [`scib`](https://github.com/theislab/scib), the package used to evaluate batch correction methods. We used the older version of scib (package: scIB), if you install the latest version, please change the following line in eval.py:
```Python
from moco.evaluation_scib_oldVersion import scib_process
====>
from moco.evaluation_scib import scib_process
``` 

## Results for a quick overview 
We released our outputs and results in [`zenodo`](https://zenodo.org/record/7136754). For a quick overview on the outputs, please visit [`plot_res.ipynb`](plot_res.ipynb)
