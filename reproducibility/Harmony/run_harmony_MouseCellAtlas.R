
# Author : Kok Siong Ang 
# Date : 17/09/2019
# Proj : Run MNN Correct pipeline

########################
#load packages

library(ggplot2)
library(cowplot)
library(Seurat)  
library(batchelor)
library(Rtsne)
library(scales)
library(glue)


rm(list=ls())


########################
#settings

filter_genes = F
filter_cells = F
normData = T
Datascaling = T
regressUMI = F
min_cells = 3
min_genes = 0
norm_method = "LogNormalize"
scale_factor = 10000
# b_x_low_cutoff = 0.0125
# b_x_high_cutoff = 3
# b_y_cutoff = 0.5
numVG = 2000
npcs = 50
k.weight = 100   # seurat default
visualize = T
outfile_prefix = "dataset2" # MouseCellAtlas
save_obj = F

src_dir = "/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Script/cusm_benchmark/harmony/"
working_dir = glue("/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Output/{outfile_prefix}/")
read_dir = glue("/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Data/{outfile_prefix}/")

expr_filename = 'filtered_total_batch1_seqwell_batch2_10x.txt'
metadata_filename = 'filtered_total_sample_ext_organ_celltype_batch.txt'

batch_label = "batchlb"
celltype_label = "CellType"

########################
# read data 

expr_mat = read.table(file = paste0(read_dir,expr_filename),sep="\t",header=T,row.names=1,check.names = F)
metadata = read.table(file = paste0(read_dir,metadata_filename),sep="\t",header=T,row.names=1,check.names = F)


colnames(metadata)[colnames(metadata) == 'ct'] = 'CellType'
expr_mat <- expr_mat[, rownames(metadata)]

source(paste0(src_dir,'call_harmony.R'))

##########################
# process
harmony_emb = harmony_process(expr_mat, metadata, 
                      filter_genes = filter_genes, filter_cells = filter_cells,
                      normData = normData, Datascaling = Datascaling, regressUMI = regressUMI, 
                      min_cells = min_cells, min_genes = min_genes, 
                      norm_method = norm_method, scale_factor = scale_factor, 
                      numVG = numVG, npcs = npcs, k.weight=k.weight, 
                      batch_label = batch_label, celltype_label = celltype_label,
                      # plot_raw=TRUE,
                      outfilename_prefix=glue('{outfile_prefix}'))

write.table(harmony_emb, 
            file=paste0(working_dir, glue("{outfile_prefix}_harmony_pca.csv")), quote=F, sep='\t', row.names = T, col.names = NA)


