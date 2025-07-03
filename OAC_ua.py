import scanpy as sc
import anndata as ad
import numpy as np
import bbknn
import pandas as pd
import glob
import os
import infercnvpy as cnv

### load in h5ad objects
OAC_ua = sc.read("/home/itrg/University/RPC/sc_data/OAC_ua.h5ad")
### Set parameters here 
npcs = 20
hvgs = 2500
res = 0.7
mindist = 0.1
spr = 0.5

### Normalisation  
# remove all NaNs
OAC_ua.X = np.nan_to_num(OAC_ua.X)
# save raw count data 
# Saving count data
OAC_ua.layers["counts"] = OAC_ua.X.copy()
# layers counts is completely raw data
# Normalizing to median total counts
sc.pp.normalize_total(OAC_ua)
# Logarithmize the data
sc.pp.log1p(OAC_ua)
# freeze the state in '.raw'
OAC_ua.raw = OAC_ua
# raw is normalised counts not filtered

### Feature selection 
sc.pp.highly_variable_genes(OAC_ua, n_top_genes=hvgs, batch_key="sample")
sc.pl.highly_variable_genes(OAC_ua)
# actually filtering 
OAC_ua = OAC_ua[:, OAC_ua.var.highly_variable]
# to return to the anndata object prefiltering this is the rescue code
# adata.raw.to_adata()

### Dimensionalilty reduction (PCA)
sc.pp.scale(OAC_ua)
sc.tl.pca(OAC_ua)
sc.pl.pca_variance_ratio(OAC_ua, n_pcs=npcs, log=True)
sc.pl.pca(
    OAC_ua,
    color=["sample", "sample", "pct_counts_mt", "pct_counts_mt"],
    dimensions=[(0, 1), (2, 3), (0, 1), (2, 3)],
    ncols=2,
    size=20,
)

### BBKNN 
# batch correction 
# bbknn
sc.external.pp.bbknn(OAC_ua, batch_key="sample", n_pcs=npcs)

### PCA plot post batch correction 
sc.pl.pca(OAC_ua, color=["sample"], dimensions=[(0, 1)], ncols=2, size=20,)
# Store the PCA coordinates in a new column in .obs
OAC_ua.obs[['pca1', 'pca2']] = OAC_ua.obsm['X_pca'][:, :2]
### removing outliers from PC2
pc2_scores = OAC_ua.obs['pca2']
sorted_indices = pc2_scores.argsort()
indices_to_keep = sorted_indices[20:-20]
# Filter in place
OAC_ua = OAC_ua[indices_to_keep, :].copy()

### remove the outliers from PC1
outlier_cell_1 = 'TGGGATTACAGG'  # replace with real cell ID
OAC_ua = OAC_ua[OAC_ua.obs_names != outlier_cell_1].copy()
#OAC_ua = OAC_ua[OAC_ua.obs_names != outlier_cell_2].copy()
# replot PCA
sc.pl.pca(OAC_ua, color=["sample"], dimensions=[(0, 1)], ncols=2, size=20,)
sc.pl.pca(OAC_ua, color=["total_counts"], dimensions=[(0, 1)], ncols=2, size=20,)
sc.pl.pca(OAC_ua, color=["n_genes_by_counts"], dimensions=[(0, 1)], ncols=2, size=20,)
sc.pl.pca(OAC_ua, color=["pct_counts_mt"], dimensions=[(0, 1)], ncols=2, size=20,)

### Nearest neighbour graph
sc.pp.neighbors(OAC_ua, n_neighbors=8, use_rep='X_pca') ### this is not required if running bbknn
sc.tl.umap(OAC_ua, min_dist=mindist, spread=spr, )
sc.pl.umap(
    OAC_ua,
    color="sample",
    # Setting a smaller point size to get prevent overlap
    size=20,
)









































