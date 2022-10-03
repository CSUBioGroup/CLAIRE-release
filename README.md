# CLAIRE
contrastive learning-based batch correction framework for better balance between batch mixing and preservation of cellular heterogeneity

## Installation

Ensure Pytorch is installed in your python environment (our test version: pytorch 1.7.1). Then installing the dependencies:
```
pip install -r requirements.txt
```

## Datasets
All datasets used in our paper can be found in [`zenodo`](https://zenodo.org/record/7136754)

## Usage
Run ```sh train_script.sh``` to reproduce our outputs. 
Then, run ```python eval.py [dataset_name]``` to evaluate the outputs. 

Note that there are several versions of [`scib`](https://github.com/theislab/scib), the package used to evaluate batch correction methods. We used the older version of scib (package: scIB), if you install the latest version, please change the following line in eval.py:
```
from moco.evaluation_scib_oldVersion import scib_process
====>
from moco.evaluation_scib import scib_process
``` 


We released our outputs and results in [`zenodo`](https://zenodo.org/record/7136754). For a quick overview on the outputs, please visit [`plot_res.ipynb`](plot_res.ipynb)
