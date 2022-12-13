rm(list=ls())
library(Seurat)  
library(batchelor)
library(Rtsne)
library(scales)
library(glue)
library(SMNN)
library(iSMNN)


# setwd('/home/yxh/gitrepo/clMining/scmoco/moco')
# source('find_anchors_utils.R')

# settings
filter_genes = F
filter_cells = F
normData = T
Datascaling = T
regressUMI = F
min_cells = 0
min_genes = 0
norm_method = "LogNormalize"
scale_factor = 10000

numVG = 2000
npcs = 30
batch_label = "batch_id"
celltype_label = "CellType"


########################
# MouseCellAtlas Dataset
#+++++++++++++++++++++++
dname = "dataset2"
read_dir = '/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Script/sapling/GLOBE/data/MouseCellAtlas/'
expr_filename = 'filtered_total_batch1_seqwell_batch2_10x.txt'
metadata_filename = 'filtered_total_sample_ext_organ_celltype_batch.txt'
working_dir = '/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Output/dataset2/'

# reading
expr_mat = read.table(file = paste0(read_dir,expr_filename),sep="\t",header=T,row.names=1,check.names = F)
metadata = read.table(file = paste0(read_dir,metadata_filename),sep="\t",header=T,row.names=1,check.names = F)

colnames(metadata)[colnames(metadata) == 'batchlb'] <- 'batch_id'
colnames(metadata)[colnames(metadata) == 'ct'] = 'CellType'

expr_mat <- expr_mat[, rownames(metadata)]


########################
#  ImmHuman Dataset
#+++++++++++++++++++++++
dname = "imm_human"
read_dir = '/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Script/sapling/GLOBE/data/ImmHuman/'
working_dir = glue('/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Output/{dname}/')

expr_filename = 'imm_human.txt'

batch_label = "batch_id"
celltype_label = "CellType"
########################
# read data 

expr_mat <- read.table(file = paste0(read_dir,expr_filename),sep=",",header=F,check.names = F)
cnames = read.table(file=paste0(read_dir, 'imm_human_cname.txt'), sep='\t', header=T)
gnames = read.table(file=paste0(read_dir, 'imm_human_gname.txt'), sep='\t', header=T)

colnames(expr_mat) = cnames$cnames
rownames(expr_mat) = gnames$gnames

metadata <- read.table(file = paste0(read_dir, 'imm_human_meta.txt'),sep=",",header=T,row.names=1,check.names = F)

colnames(metadata)[colnames(metadata) == 'batch'] <- 'batch_id'
colnames(metadata)[colnames(metadata) == 'final_annotation'] <- 'CellType'

expr_mat <- expr_mat[, rownames(metadata)]


########################
#  Pancreas Dataset
#+++++++++++++++++++++++
dname = "dataset4"
read_dir = '/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Script/sapling/GLOBE/data/Pancreas/'
working_dir = glue('/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Output/{dname}/')

expr_mat_filename = "myData_pancreatic_5batches.txt"
metadata_filename = "mySample_pancreatic_5batches.txt"

batch_label = "batch_id"
celltype_label = "CellType"

# read data 
expr_mat <- read.table(file = paste0(read_dir,expr_mat_filename),sep="\t",header=T,row.names=1,check.names = F)
metadata <- read.table(file = paste0(read_dir,metadata_filename),sep="\t",header=T,row.names=1,check.names = F)

colnames(metadata)[colnames(metadata) == 'batchlb'] <- 'batch_id'
colnames(metadata)[colnames(metadata) == 'celltype'] <- "CellType"

expr_mat <- expr_mat[, rownames(metadata)]

########################
#  PBMC Dataset
#+++++++++++++++++++++++

dname = "dataset5"
read_dir = '/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Script/sapling/GLOBE/data/PBMC/'
working_dir = glue('/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Output/{dname}/')

b1_exprs_filename = "b1_exprs.txt"
b2_exprs_filename = "b2_exprs.txt"
b1_celltype_filename = "b1_celltype.txt"
b2_celltype_filename = "b2_celltype.txt"

batch_label = "batch_id"
celltype_label = "CellType"

# read data 

b1_exprs <- read.table(file = paste0(read_dir,b1_exprs_filename),sep="\t",header=T,row.names=1,check.names = F)
b2_exprs <- read.table(file = paste0(read_dir,b2_exprs_filename),sep="\t",header=T,row.names=1,check.names = F)
b1_celltype <- read.table(file = paste0(read_dir,b1_celltype_filename),sep="\t",header=T,row.names=1,check.names = F)
b2_celltype <- read.table(file = paste0(read_dir,b2_celltype_filename),sep="\t",header=T,row.names=1,check.names = F)

# b1_celltype$cell <- rownames(b1_celltype)
b1_celltype <- b1_celltype[colnames(b1_exprs),]
# b2_celltype$cell <- rownames(b2_celltype)
b2_celltype <- b2_celltype[colnames(b2_exprs),]

b1_metadata <- as.data.frame(b1_celltype)
b2_metadata <- as.data.frame(b2_celltype)

b1_metadata$batch_id <- 'Batch1'
b2_metadata$batch_id <- 'Batch2'

expr_mat = cbind(b1_exprs,b2_exprs)
metadata = rbind(b1_metadata, b2_metadata)

expr_mat <- expr_mat[, rownames(metadata)]

########################
#  Lung Dataset
#+++++++++++++++++++++++
dname = "lung"
read_dir = '/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Script/sapling/GLOBE/data/Lung/'
working_dir = glue('/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Output/{dname}/')

expr_filename = 'lung.txt'

batch_label = "batch_id"
celltype_label = "CellType"
########################
# read data 

expr_mat <- read.table(file = paste0(read_dir,expr_filename),sep=",",header=F,check.names = F)
cnames = read.table(file=paste0(read_dir, 'lung_cname.txt'), sep='\t', header=T)
gnames = read.table(file=paste0(read_dir, 'lung_gname.txt'), sep='\t', header=T)

colnames(expr_mat) = cnames$cnames
rownames(expr_mat) = gnames$gnames

metadata <- read.table(file = paste0(read_dir, 'lung_meta.txt'),sep=",",header=T,row.names=1,check.names = F)

colnames(metadata)[colnames(metadata) == 'batch'] <- 'batch_id'
colnames(metadata)[colnames(metadata) == 'cell_type'] <- 'CellType'

expr_mat <- expr_mat[, rownames(metadata)]



########################
#  Muris Dataset
#+++++++++++++++++++++++
dname = "muris"
read_dir = '/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Script/sapling/GLOBE/data/Muris/'
working_dir = glue('/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Output/{dname}/')

expr_filename = 'muris_subsample_filter.txt'

batch_label = "batch_id"
celltype_label = "CellType"
########################
# read data 

expr_mat <- read.table(file = paste0(read_dir,expr_filename),sep=",",header=F,check.names = F)
cnames = read.table(file=paste0(read_dir, 'muris_subsample_filter_cname.txt'), sep='\t', header=T)
gnames = read.table(file=paste0(read_dir, 'muris_subsample_filter_gname.txt'), sep='\t', header=T)

colnames(expr_mat) = cnames$cnames
rownames(expr_mat) = gnames$gnames

metadata <- read.table(file = paste0(read_dir, 'muris_subsample_filter_meta.txt'),sep=",",header=T,row.names=1,check.names = F)

# colnames(metadata)[colnames(metadata) == 'batch'] <- 'batch_id'
colnames(metadata)[colnames(metadata) == 'cell_ontology_class'] <- 'CellType'

metadata[metadata$batch==0, 'batch_id'] = 'batch0' 
metadata[metadata$batch==1, 'batch_id'] = 'batch1'

expr_mat <- expr_mat[, rownames(metadata)]


#### ===================================shared operations================================
# k_filter: larger=> more anchors, more iterations
# k_weight: larger=> more global translation
# k_anchor: larger=> more anchors, more iterations 

source('./ismnn_utils.R')
ismm_results = ismnn(expr_mat, metadata, 
                    matched.clusters=NULL,
                    iterations = 5, k_anchor=5, k_filter=100, k_scores=100, k_weight=100,
                    filter_genes = F, filter_cells = F,
                    normData = T, Datascaling = T, regressUMI = F, 
                    min_cells = min_cells, min_genes = min_genes, norm_method = "LogNormalize", scale_factor = scale_factor, 
                    # b_x_low_cutoff = 0.0125, b_x_high_cutoff = 3, b_y_cutoff = 0.5, 
                    numVG = numVG, npcs = npcs,
                    batch_label = batch_label, celltype_label = "CellType",
                    outfilename_prefix=dname)


# UMAP results
ismm_umap = data.frame(Embeddings(ismm_results, reduction='umap'))
dim(ismnn_umap)

colnames(ismnn_umap) = paste0('UMAP', 1:dim(ismnn_umap)[2])

ismnn_umap[['batchlb']] = ismm_results@meta.data$batch_id
ismnn_umap[['CellType']] = ismm_results@meta.data$CellType

write.table(ismnn_umap, 
        file=paste0(working_dir, glue('{dname}_ismnn_umap.csv')), quote=F, sep='\t', row.names=T, col.names=NA)

# PCA results
ismm_pca = data.frame(Embeddings(ismm_results, reduction='pca'))
dim(ismm_pca)

colnames(ismm_pca) = paste0('PC', 1:dim(ismm_pca)[2])

ismm_pca[['batchlb']] = ismm_results@meta.data$batch_id
ismm_pca[['CellType']] = ismm_results@meta.data$CellType

write.table(ismm_pca, 
        file=paste0(working_dir, glue('{dname}_ismnn_pca.csv')), quote=F, sep='\t', row.names=T, col.names=NA)


# plot
p1 = DimPlot(ismm_results, reduction='umap', group.by='batch_id')
p2 = DimPlot(ismm_results, reduction='umap', group.by='CellType')

png(paste0(working_dir, dname, '_ismnn_umap_k-anc=5_k-fil=100_k-w=100_k-s=100', '.png'), width=2*1000, height=800, res=2*72)
print(plot_grid(p1, p2))
dev.off()
