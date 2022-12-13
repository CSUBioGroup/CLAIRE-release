
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
k.weight = 100
visualize = T
outfile_prefix = "lung"
save_obj = F

src_dir = "/home/gitrepo/Batch-effect-removal-benchmarking-master/Script/cusm_benchmark/seurat/"
working_dir = glue("/home/gitrepo/Batch-effect-removal-benchmarking-master/Output/{outfile_prefix}/")
read_dir = glue("/home/gitrepo/Batch-effect-removal-benchmarking-master/Data/{outfile_prefix}/")

expr_filename = 'lung.txt'

batch_label = "batchlb"
celltype_label = "CellType"
########################
# read data 

expr_mat <- read.table(file = paste0(read_dir,expr_filename),sep=",",header=F,check.names = F)
cnames = read.table(file=paste0(read_dir, 'lung_cname.txt'), sep='\t', header=T)
gnames = read.table(file=paste0(read_dir, 'lung_gname.txt'), sep='\t', header=T)

colnames(expr_mat) = cnames$cnames
rownames(expr_mat) = gnames$gnames

metadata <- read.table(file = paste0(read_dir, 'lung_meta.txt'),sep=",",header=T,row.names=1,check.names = F)

colnames(metadata)[colnames(metadata) == 'batch'] <- 'batchlb'
colnames(metadata)[colnames(metadata) == 'cell_type'] <- 'CellType'

expr_mat <- expr_mat[, rownames(metadata)]

source(paste0(src_dir,'call_seurat.R'))

##########################
# process
# anchors_return = finding_anchors(expr_mat, metadata, 
#                           filter_genes = filter_genes, filter_cells = filter_cells,
#                           normData = normData, Datascaling = Datascaling, regressUMI = regressUMI, 
#                           min_cells = min_cells, min_genes = min_genes, 
#                           norm_method = norm_method, scale_factor = scale_factor, 
#                           numVG = numVG, npcs = npcs, k.weight=k.weight, 
#                           batch_label = batch_label, celltype_label = celltype_label,
#                           # plot_raw=TRUE,
#                           outfilename_prefix=glue('{outfile_prefix}'))

# anchors = anchors_return$anchors
# bso.list = anchors_return$Xlist

# head(bso.list[[1]]@meta.data)

# df_anchor = anchors@anchors
# df_anchor$name1 = 'empty'
# df_anchor$name2 = 'empty'

# for (i in 1:dim(df_anchor)[1]){
#     c1 = df_anchor[i, 'cell1']
#     c2 = df_anchor[i, 'cell2']
#     d1 = df_anchor[i, 'dataset1']
#     d2 = df_anchor[i, 'dataset2']
#     df_anchor[i, 'name1'] = colnames(bso.list[[d1]])[c1]
#     df_anchor[i, 'name2'] = colnames(bso.list[[d2]])[c2]
# }


# write.table(df_anchor, file=paste0(read_dir, glue("{outfile_prefix}_seuratAnchors.csv")), quote=F, sep=',', row.names = T, col.names = NA)



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
dim(seurat_pca)

colnames(seurat_pca) = paste0('PC', 1:npcs)

seurat_pca[[batch_label]] = bso@meta.data$batchlb
seurat_pca[[celltype_label]] = bso@meta.data$CellType

write.table(seurat_pca, 
            file=paste0(working_dir, glue("{outfile_prefix}_seurat_pca.csv")), quote=F, sep='\t', row.names = T, col.names = NA)


