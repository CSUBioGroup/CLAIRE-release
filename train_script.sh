# mca
python main.py --dname 'MouseCellAtlas' --n_repeat 3 --moco_k 2048 --moco_m 0.999 --moco_t 0.07 \
                    --block_level 1 --lat_dim 128 --symmetric True \
                    --select_hvg 2000 \
                    --knn 10 --alpha 0.5 --augment_set 'int' \
                    --anchor_schedule 4 --fltr 'gmm' --yita 0.5 \
                    --lr 1e-4 --optim Adam --weight_decay 1e-5 --epochs 120 --batch_size 256\
                    --save_freq 10 --start_epoch 0 --workers 6 --init 'uniform' \
                    --visualize_ckpts 10 20 40 80 120

# # pbmc
python main.py --dname 'PBMC' --n_repeat 3 --moco_k 2048 --moco_m 0.999 --moco_t 0.07 \
                    --block_level 1 --lat_dim 128 --symmetric True \
                    --select_hvg 2000 \
                    --knn 10 --alpha 0.5 --augment_set 'int' \
                    --lr 1e-4 --adjustLr --schedule 10 --optim Adam --weight_decay 1e-5 --epochs 120 --batch_size 256\
                    --anchor_schedule 10 --fltr 'gmm' --yita 0.5 \
                    --save_freq 10 --start_epoch 0 --workers 6 --init 'uniform' \
                    --visualize_ckpts 10 20 40 80 120

# pancreas
python main.py --dname 'Pancreas' --n_repeat 3 --moco_k 2048 --moco_m 0.999 --moco_t 0.07 \
                    --block_level 1 --lat_dim 128 --symmetric True \
                    --select_hvg 2000 \
                    --knn 10 --alpha 0.5 --augment_set 'int' \
                    --lr 1e-4 --optim Adam --weight_decay 1e-5 --epochs 120 --batch_size 256\
                    --anchor_schedule 2 --fltr 'gmm' --yita 0.5 \
                    --save_freq 10 --start_epoch 0 --workers 6 --init 'uniform' \
                    --visualize_ckpts 10 20 40 80 120   

# immune
python main.py --dname 'ImmHuman' --n_repeat 3 --moco_k 2048 --moco_m 0.999 --moco_t 0.07 \
                    --block_level 1 --lat_dim 128 --symmetric True \
                    --select_hvg 2000 \
                    --knn 10 --alpha 0.5 --augment_set 'int' \
                    --lr 1e-4 --optim Adam --weight_decay 1e-5 --epochs 120 --batch_size 256\
                    --anchor_schedule 2 --fltr 'gmm' --yita 0.5 \
                    --save_freq 10 --start_epoch 0 --workers 6 --init 'uniform' \
                    --visualize_ckpts 10 20 40 80 120 

# lung
python main.py --dname 'Lung' --n_repeat 3 --moco_k 2048 --moco_m 0.999 --moco_t 0.07 \
                    --block_level 1 --lat_dim 128 --symmetric True \
                    --select_hvg 2000 \
                    --knn 10 --alpha 0.5 --augment_set 'int' \
                    --lr 1e-4 --optim Adam --weight_decay 1e-5 --epochs 120 --batch_size 256\
                    --anchor_schedule 5 --fltr 'gmm' --yita 0.5 \
                    --save_freq 10 --start_epoch 0 --workers 6 --init 'uniform' \
                    --visualize_ckpts 10 20 40 80 120

# muris
python main.py --dname 'Muris' --n_repeat 3 --moco_k 2048 --moco_m 0.999 --moco_t 0.07 \
                    --block_level 1 --lat_dim 128 --symmetric True \
                    --select_hvg 5000 \
                    --knn 10 --alpha 0.5 --augment_set 'int' \
                    --lr 1e-4 --adjustLr --schedule 10 --optim Adam --weight_decay 1e-5 --epochs 120 --batch_size 256\
                    --anchor_schedule 10 --fltr 'gmm' --yita 0.6 \
                    --save_freq 10 --start_epoch 0 --workers 6 --init 'uniform' \
                    --visualize_ckpts 10 20 40 80 120