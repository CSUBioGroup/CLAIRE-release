
# library(Seurat)  # Seurat >= 3
# library(SMNN)
# library(iSMNN)
ismnn <- function(expr_mat_mnn, metadata, 
                  matched.clusters = NULL,
                  iterations = 5, k_anchor=5, k_filter=10, k_scores=10, k_weight=10,
                  filter_genes = T, filter_cells = T,
                  normData = T, Datascaling = T, regressUMI = F, 
                  min_cells = 10, min_genes = 300, norm_method = "LogNormalize", scale_factor = 10000, 
                  # b_x_low_cutoff = 0.0125, b_x_high_cutoff = 3, b_y_cutoff = 0.5, 
                  numVG = 2000, npcs = 30,
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


  b_seurat <- CreateSeuratObject(counts = expr_mat_mnn, meta.data = metadata, project = "ismnn_merge", 
                                 min.cells = min_cells, min.genes = min_genes)

  # split object by batch
  merge.list <- SplitObject(b_seurat, split.by = batch_label)

  t1 = Sys.time()
  # normalize and identify variable features for each dataset independently
  merge.list <- lapply(X = merge.list, FUN = function(x) {
      x <- NormalizeData(x)
      x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = numVG)
  })

  # batch.cluster.labels, though i don't understand what the fuck it's
  batch.cluster.labels <- lapply(X = merge.list, FUN = function(x) {
      x <- x@meta.data$CellType
  })

  for (i in 1:length(batch.cluster.labels)){
      names(batch.cluster.labels[[i]])=colnames(merge.list[[i]])
  }

  tl_bl = table(metadata$CellType, metadata$batch_id)
  share_tl = rowSums(tl_bl>0) == dim(tl_bl)[2]   # shared clusters
  min_tl   = apply(tl_bl, 1, FUN=min)

  if (is.null(matched.clusters)){
    matched.clusters = c()
    for (ti in rownames(tl_bl)){
      if (share_tl[ti] & (min_tl[ti]>=k_weight)){
        matched.clusters = c(matched.clusters, ti)
      }
    }
  }

  if (length(matched.clusters) == 0){
    print('Empty cell types intersection')
    return ('Shit')
  }
  print('Shared Clusters among all batches')
  print(matched.clusters)



  # for (ti in rownames(tl_bl)){
  #   sum(tl_bl[ti, ] > 0) == dim(tl_bl)[2]
  # }
  # matched.clusters = Reduce(intersect, batch.cluster.labels)

  # ismnn
  corrected.results <- iSMNN(object.list = merge.list, batch.cluster.labels = batch.cluster.labels, matched.clusters = matched.clusters,
                            strategy = "Short.run", iterations = 5, dims = 1:20, npcs = 30, 
                            k.anchor=k_anchor, k.filter = k_filter, k.score = k_scores, k.weight=k_weight)

  #==================TO-DO===================
  # transform the dataset.id in bso.anchors to 'dataset.name'
  t2 = Sys.time()

  print('========================================')
  print(t2 - t1)


  return (corrected.results)
}
