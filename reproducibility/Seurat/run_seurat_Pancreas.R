
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
npcs = 30
k.weight = 100
visualize = T
outfile_prefix = "dataset4"  # Pancreas
save_obj = F

src_dir = "/home/gitrepo/Batch-effect-removal-benchmarking-master/Script/cusm_benchmark/seurat/"
working_dir = glue("/home/gitrepo/Batch-effect-removal-benchmarking-master/Output/{outfile_prefix}/")
read_dir = glue("/home/gitrepo/Batch-effect-removal-benchmarking-master/Data/{outfile_prefix}/")


expr_mat_filename = "myData_pancreatic_5batches.txt"
metadata_filename = "mySample_pancreatic_5batches.txt"

batch_label = "batchlb"
celltype_label = "CellType"

########################
# read data 
expr_mat <- read.table(file = paste0(read_dir,expr_mat_filename),sep="\t",header=T,row.names=1,check.names = F)
metadata <- read.table(file = paste0(read_dir,metadata_filename),sep="\t",header=T,row.names=1,check.names = F)

colnames(metadata)[colnames(metadata) == 'celltype'] <- "CellType"

expr_mat <- expr_mat[, rownames(metadata)]

source(paste0(src_dir,'call_seurat.R'))

##########################
# process

# anchors = finding_anchors(expr_mat, metadata, 
#                           filter_genes = filter_genes, filter_cells = filter_cells,
#                           normData = normData, Datascaling = Datascaling, regressUMI = regressUMI, 
#                           min_cells = min_cells, min_genes = min_genes, 
#                           norm_method = norm_method, scale_factor = scale_factor, 
#                           numVG = numVG, npcs = npcs, k.weight=k.weight, 
#                           batch_label = batch_label, celltype_label = celltype_label,
#                           # plot_raw=TRUE,
#                           outfilename_prefix=glue('{outfile_prefix}'))

# write.table(anchors@anchors, file=paste0(read_dir, glue("{outfile_prefix}_seuratAnchors.csv")), quote=F, sep=',', row.names = T, col.names = NA)


bso = seurat_process(expr_mat, metadata, 
                      filter_genes = filter_genes, filter_cells = filter_cells,
                      normData = normData, Datascaling = Datascaling, regressUMI = regressUMI, 
                      min_cells = min_cells, min_genes = min_genes, 
                      norm_method = norm_method, scale_factor = scale_factor, 
                      numVG = numVG, npcs = npcs, k.weight=k.weight, 
                      batch_label = batch_label, celltype_label = celltype_label,
                      # plot_raw=TRUE,
                      outfilename_prefix=glue('{outfile_prefix}'))

seurat_pca = data.frame(Embeddings(bso, reduction='pca'))
# dim(seurat_pca)

colnames(seurat_pca) = paste0('PC', 1:npcs)

seurat_pca[[batch_label]] = bso@meta.data$batchlb
seurat_pca[[celltype_label]] = bso@meta.data$CellType

write.table(seurat_pca, 
            file=paste0(working_dir, glue("{outfile_prefix}_seurat_pca.csv")), quote=F, sep='\t', row.names = T, col.names = NA)


