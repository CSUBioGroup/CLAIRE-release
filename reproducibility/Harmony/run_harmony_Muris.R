
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
numVG = 5000
npcs = 50
k.weight = 100
visualize = T
outfile_prefix = "muris"
save_obj = F


src_dir = "/home/gitrepo/Batch-effect-removal-benchmarking-master/Script/cusm_benchmark/harmony/"
working_dir = glue("/home/gitrepo/Batch-effect-removal-benchmarking-master/Output/{outfile_prefix}/")
read_dir = glue("/home/gitrepo/Batch-effect-removal-benchmarking-master/Data/{outfile_prefix}/")


expr_filename = 'muris_subsample_filter.txt'

batch_label = "batchlb"
celltype_label = "CellType"
########################
# read data 

expr_mat <- read.table(file = paste0(read_dir,expr_filename),sep=",",header=F,check.names = F)
cnames = read.table(file=paste0(read_dir, 'muris_subsample_filter_cname.txt'), sep='\t', header=T)
gnames = read.table(file=paste0(read_dir, 'muris_subsample_filter_gname.txt'), sep='\t', header=T)

colnames(expr_mat) = cnames$cnames
rownames(expr_mat) = gnames$gnames

metadata <- read.table(file = paste0(read_dir, 'muris_subsample_filter_meta.txt'),sep=",",header=T,row.names=1,check.names = F)

colnames(metadata)[colnames(metadata) == 'batch'] <- 'batchlb'
colnames(metadata)[colnames(metadata) == 'cell_ontology_class'] <- 'CellType'

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


