
seurat_process <- function(expr_mat_mnn, metadata, 
                            filter_genes = T, filter_cells = T,
                            normData = T, Datascaling = T, regressUMI = F, 
                            min_cells = 10, min_genes = 300, norm_method = "LogNormalize", scale_factor = 10000, 
                            b_x_low_cutoff = 0.0125, b_x_high_cutoff = 3, b_y_cutoff = 0.5, 
                            numVG = 300, npcs = 30, k.weight=100,
                            batch_label = "batchlb", celltype_label = "CellType",
                            outfilename_prefix='simLinear_10_batches')
{

  ##########################################################
  # preprocessing

  if(filter_genes == F) {
    min_cells = 0
  }
  if(filter_cells == F) {
    min_genes = 0
  }

  b_seurat <- CreateSeuratObject(counts = expr_mat_mnn, meta.data = metadata, project = "seurat_benchmark", 
                                 min.cells = min_cells, min.genes = min_genes)

  # split object by batch
  bso.list <- SplitObject(b_seurat, split.by = batch_label)

  t1 = Sys.time()
  # normalize and identify variable features for each dataset independently
  bso.list <- lapply(X = bso.list, FUN = function(x) {
      x <- NormalizeData(x)
      x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = numVG)
  })

  # take the share hvg features, and subsets
  features <- SelectIntegrationFeatures(object.list = bso.list)

  bso.anchors <- FindIntegrationAnchors(object.list = bso.list, anchor.features = features)
  bso <- IntegrateData(anchorset = bso.anchors, k.weight=k.weight)   # for some reason, 'k.weight' should be set smaller than 100
  DefaultAssay(bso) <- "integrated"

  t2 = Sys.time()
  print('=========================================')
  print(t2-t1)

  # return (bso)
  bso <- ScaleData(bso, verbose = FALSE)
  # bso <- RunPCA(bso, npcs = npcs, verbose = FALSE)
  # bso <- RunUMAP(bso, reduction = "pca", dims = 1:npcs)

  # p1 <- DimPlot(bso, reduction = "umap", group.by = batch_label)
  # p2 <- DimPlot(bso, reduction = "umap", group.by = celltype_label)
  # # p1 + p2

  # png(paste0(working_dir,outfilename_prefix,'_seurat_umap',".png"),width = 2*1000, height = 800, res = 2*72)
  # print(plot_grid(p1, p2))
  # dev.off()

  return (bso)
}

finding_anchors <- function(expr_mat_mnn, metadata, 
                            filter_genes = T, filter_cells = T,
                            normData = T, Datascaling = T, regressUMI = F, 
                            min_cells = 10, min_genes = 300, norm_method = "LogNormalize", scale_factor = 10000, 
                            b_x_low_cutoff = 0.0125, b_x_high_cutoff = 3, b_y_cutoff = 0.5, 
                            numVG = 300, npcs = 30, k.weight=100,
                            batch_label = "batchlb", celltype_label = "CellType",
                            outfilename_prefix='simLinear_10_batches')
{

  ##########################################################
  # preprocessing

  if(filter_genes == F) {
    min_cells = 0
  }
  if(filter_cells == F) {
    min_genes = 0
  }

  b_seurat <- CreateSeuratObject(counts = expr_mat_mnn, meta.data = metadata, project = "seurat_benchmark", 
                                 min.cells = min_cells, min.genes = min_genes)

  # split object by batch
  bso.list <- SplitObject(b_seurat, split.by = batch_label)

  # normalize and identify variable features for each dataset independently
  bso.list <- lapply(X = bso.list, FUN = function(x) {
      x <- NormalizeData(x)
      x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = numVG)
  })

  # take the share hvg features, and subsets
  features <- SelectIntegrationFeatures(object.list = bso.list)

  bso.anchors <- FindIntegrationAnchors(object.list = bso.list, anchor.features = features)

  #==================TO-DO===================
  # transform the dataset.id in bso.anchors to 'dataset.name'

  return (list(anchors=bso.anchors, Xlist=bso.list))
}