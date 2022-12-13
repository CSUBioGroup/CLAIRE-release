import os
import sys
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from os.path import join
cur_dir = Path(os.getcwd())
sys.path.append(str(cur_dir.parent.parent.absolute()))

from moco.kbet import calculate_kbet
from moco.utils import py_read_data, load_meta_txt
from moco.evaluation import scib_process 

def scib_eval(ada, batch_key='batchlb', label_key='CellType'):
    tmp_res = scib_process(ada,
                                batch_key=batch_key,
                                label_key=label_key,
                                silhouette_=True
                                )
    # calc kbet
    kt = calculate_kbet(ada,
            use_rep='X',
            batch_col=batch_key,
            n_neighbors=15,         # ensure the n larger than the one used before
            calc_knn=True,
            n_jobs=10,
        )[2]
    tmp_res.loc['kBET'] = kt  # 'loc' attention,
    return tmp_res
