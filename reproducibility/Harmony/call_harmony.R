library(harmony)
library(Seurat)

harmony_process <- function(expr_mat_mnn, metadata, 
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

  features <- SelectIntegrationFeatures(object.list = bso.list)

  b_seurat <-  ScaleData(b_seurat, verbose = FALSE, features=features)
  b_seurat <- RunPCA(b_seurat, npcs = npcs, verbose = FALSE, features=features)

  # ===========================
  # some stupid issues
  # b_seurat <- b_seurat %>% 
  #                 RunHarmony(batch_label, plot_convergence = TRUE)
  # ===========================

  embedding <- Seurat::Embeddings(b_seurat, reduction = "pca")
  metavars_df <- Seurat::FetchData(b_seurat, batch_label)

  # default params from harmony
  theta = NULL
  lambda = NULL
  sigma = 0.1
  nclust = NULL
  tau = 0
  block.size = 0.05
  max.iter.harmony = 10
  max.iter.cluster = 20
  epsilon.cluster = 1e-5
  epsilon.harmony = 1e-4
  plot_convergence = FALSE
  verbose = TRUE
  reference_values = NULL

  harmonyEmbed <- HarmonyMatrix(
      embedding,
      metavars_df,
      batch_label,
      FALSE,
      0,
      theta,
      lambda,
      sigma,
      nclust,
      tau,
      block.size,
      max.iter.harmony,
      max.iter.cluster,
      epsilon.cluster,
      epsilon.harmony,
      plot_convergence,
      FALSE,
      verbose,
      reference_values
  )
                  
  return (harmonyEmbed)
}