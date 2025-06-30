#02/06/2025
### data analysis of metastatic LN

import scanpy as sc
import anndata as ad
import numpy as np
import bbknn
import pandas as pd
import glob
import os

### load in h5ad objects
OAC_pt = sc.read("/home/itrg/University/RPC/sc_data/OAC_pt.h5ad")

### Set parameters here 
npcs = 20
hvgs = 2500
res = 0.7
mindist = 0.3

### Normalisation  
# remove all NaNs
OAC_pt.X = np.nan_to_num(OAC_pt.X)
# save raw count data 
# Saving count data
OAC_pt.layers["counts"] = OAC_pt.X.copy()
# Normalizing to median total counts
sc.pp.normalize_total(OAC_pt)
# Logarithmize the data
sc.pp.log1p(OAC_pt)
# freeze the state in '.raw'
OAC_pt.raw = OAC_pt

### Feature selection 
sc.pp.highly_variable_genes(OAC_pt, n_top_genes=hvgs, batch_key="sample")
sc.pl.highly_variable_genes(OAC_pt)
# actually filtering 
OAC_pt = OAC_pt[:, OAC_pt.var.highly_variable]
# to return to the anndata object prefiltering this is the rescue code
# adata.raw.to_adata()

### Dimensionalilty reduction (PCA)
sc.pp.scale(OAC_pt)
sc.tl.pca(OAC_pt)
sc.pl.pca_variance_ratio(OAC_pt, n_pcs=npcs, log=True)
sc.pl.pca(
    OAC_pt,
    color=["sample", "sample", "pct_counts_mt", "pct_counts_mt"],
    dimensions=[(0, 1), (2, 3), (0, 1), (2, 3)],
    ncols=2,
    size=20,
)

### BBKNN 
# batch correction 
# bbknn
sc.external.pp.bbknn(OAC_pt, batch_key="sample", n_pcs=npcs)

### PCA plot post batch correction 
sc.pl.pca(OAC_pt, color=["sample"], dimensions=[(0, 1)], ncols=2, size=20,)

### Nearest neighbour graph
# sc.pp.neighbors(OAC_pt) ### this is not required if running bbknn
sc.tl.umap(OAC_pt, min_dist=mindist)
sc.pl.umap(
    OAC_pt,
    color="sample",
    # Setting a smaller point size to get prevent overlap
    size=20,
)

### clustering 
# Leiden graph-clustering method
# Using the igraph implementation and a fixed number of iterations can be significantly faster, especially for larger datasets
sc.tl.leiden(OAC_pt, resolution=res)
sc.pl.umap(OAC_pt, color=["leiden"], size=20) # change colour with palette parameter

# set marker genes 
# cell types: Cancer cells, B cell, NK cell, T cell, monocyte, dendritic cell, lymphoid cell, macrophage, mast cell, non immune cell, adipocyte, fibroblast, pericyte, smooth muscle cell, 
# non immune cell types: adipocyte, fibroblast reticular cell, CAF, pericyte, sooth muscle cell, endothelial cell, CXC12 reticular cells, capillary resident progenitor, follicular dendritic cells, marginal reticular cell, plasma cell, perivascular cell, subscapular sinus
marker_genes = {"B cell": ["CD79A", "MS4A1", "CD19"], "NK cell": ["NCAM1", "NKG7"], "T cell": ["CD3D", "CD3E"], "Monocyte": ["FCN1", "CD14"], "Dendritic cell": ["CD1C", "FCER1A", "CLEC10A"], 
"Macrophage": ["C1QA", "CD68", "MS4A7"], "Mast cell": ["TPSAB1", "TPSB2", "CPA3"], "Cancer Associated Fibroblast": ["FAP", "ACTA2", "PDPN"], "Endothelial cell": ["PECAM1", "VWF"],
"Fibroblast": ["COL1A1", "DCN", "PDGFRA"], "Pericyte": ["PDGFRB", "RGS5"], "Smooth muscle cells": ["ACTA2", "MYH11", "TAGLN"], "Fibroblast reticular cell": ["PDPN", "CCL21"],
"Follicular dendritic cell": ["CR2", "FCER2", "CR1"], "Plasma cell": ["SDC1", "CD38", "MZB1"], "Cancer cell": ["EPCAM", "CDH1"]}
# subset to only the markers found in the data
marker_genes_in_data = {}
for ct, markers in marker_genes.items():
    markers_found = []
    for marker in markers:
        if marker in OAC_pt.var.index:
            markers_found.append(marker)
    marker_genes_in_data[ct] = markers_found

marker_genes_II = ["MS4A1", "GNLY", "CD3D", "FCN1", "CD1C", "C1QA", "TPSAB1", "FAP", "PECAM1", "COL1A1", "PDGFRB", "ACTA2", "PDPN", "CR2", "SDC1", "EPCAM", "CDH1"]
sc.pl.umap(OAC_pt, color=marker_genes_II, use_raw=True)

# Dot plot table 
sc.pl.dotplot(
    OAC_pt,
    groupby="leiden",
    var_names=marker_genes,
    standard_scale="var",  # standard scale: normalize each gene to range from 0 to 1
    use_raw=True
)

### Labelled UMAP
cluster_to_celltype = {
    '0': 'Plasma cells',
    '1': 'T cells',
    '2': 'T cells',
    '3': 'Cancer cells',
    '4': 'NK cells',
    '5': 'Cancer cells',
    '6': 'Myeloid cells',
    '7': 'Plasma cells',
    '8': 'Fibroblasts',
    '9': 'B cells',
    '10': 'Endothelial cells',
    '11': 'Mast cells',
    '12': 'Cancer cells',
    '13': 'T cells',
    '14': 'Smooth Muscle cells'
    # Add more as needed
}

# Create the new column
OAC_pt.obs['cell_type'] = OAC_pt.obs['leiden'].map(cluster_to_celltype)
sc.pl.umap(OAC_pt, color=["cell_type"], size=20)

# Vionlin plot 
# showing the same as above
sc.pl.violin(OAC_pt, marker_genes_II, groupby='leiden')

# Differentially expressed genes in each cluster 
sc.tl.rank_genes_groups(
    OAC_pt, groupby="leiden", method="wilcoxon", key_added="dea_leiden"
)
# Get the result dictionary
result = OAC_pt.uns['dea_leiden']

# 'names' contains the ranked gene names per cluster
groups = result['names'].dtype.names  # cluster names

for group in groups:
    print(f"Top 10 DE genes for cluster {group}:")
    top_genes = result['names'][group][:20]
    print(top_genes)
    print("\n")
sc.pl.rank_genes_groups_dotplot(
    OAC_pt, groupby="leiden", standard_scale="var", n_genes=10, key="dea_leiden"
)

### subset and recluster the stromal cells 
# Create a Boolean mask for stromal clusters
fibroblast_mask = OAC_pt.obs['leiden'].isin(['8'])

# Use the mask to subset the AnnData object (this is key!)
fibroblast = OAC_pt[fibroblast_mask].copy()

# Now you can run Scanpy functions on this new AnnData object
sc.pp.neighbors(fibroblast)
sc.tl.leiden(fibroblast, resolution=0.5)
sc.tl.umap(fibroblast, min_dist=0.2)
sc.pl.umap(fibroblast, color=['leiden'], size= 500)

### Stromal Subset cell annotation 
fibroblast_marker_genes = {"Cancer Associated Fibroblast": ["FAP", "ACTA2", "PDPN"],
"Fibroblast": ["COL1A1", "DCN", "PDGFRA"], "Fibroblast reticular cell": ["PDPN", "CCL21"]}

# dot plot 
sc.pl.dotplot(
    fibroblast,
    groupby="leiden",
    var_names=fibroblast_marker_genes,
    standard_scale="var",  # standard scale: normalize each gene to range from 0 to 1
    use_raw=True
)

# umap plots
stromal_marker_genes_II = ["FAP", "PECAM1", "COL1A1", "PDGFRB", "ACTA2", "PDPN"]
sc.pl.umap(fibroblast, color=stromal_marker_genes_II, use_raw=True)

# Differentially expressed genes in each cluster 
sc.tl.rank_genes_groups(
    fibroblast, groupby="leiden", method="wilcoxon", key_added="dea_leiden"
)
# Get the result dictionary
result = fibroblast.uns['dea_leiden']

# 'names' contains the ranked gene names per cluster
groups = result['names'].dtype.names  # cluster names

for group in groups:
    print(f"Top 10 DE genes for cluster {group}:")
    top_genes = result['names'][group][:10]
    print(top_genes)
    print("\n")
sc.tl.dendrogram(fibroblast, groupby='leiden')
sc.pl.rank_genes_groups_dotplot(
    fibroblast, groupby="leiden", standard_scale="var", n_genes=10, key="dea_leiden"
)

### subset and recluster the cancer cells 
# Create a Boolean mask for cancer clusters
cancer_mask = OAC_pt.obs['leiden'].isin(['3', '5', '12'])

# Use the mask to subset the AnnData object (this is key!)
cancer = OAC_pt[cancer_mask].copy()

# Now you can run Scanpy functions on this new AnnData object
sc.pp.neighbors(cancer)
sc.tl.leiden(cancer, resolution=0.5)
sc.tl.umap(cancer, min_dist=0.2)
sc.pl.umap(cancer, color=['leiden'])

### Cancer Subset cell annotation 
EMT_marker_genes = {"Epithelial cell": ["EPCAM", "CDH1"], "EMT TF": ["SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1", "TWIST2"], "Mesenchymal cell": ["VIM", "FN1", "CDH2", "MMP1", "SMN1", ], "EMT-Inducing factors": ["TGFB1", "EGF", "CTNNB1", "NOTCH1", "MYC"] }

# dot plot 
sc.pl.dotplot(
    cancer,
    groupby="leiden",
    var_names=EMT_marker_genes,
    standard_scale="var",  # standard scale: normalize each gene to range from 0 to 1
    use_raw=True
)

# umap plots
cancer_marker_genes_II = ["EPCAM", "CDH1", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1", "TWIST2", "VIM", "FN1", "CDH2", "MMP1", "SMN1", "TGFB1", "CTNNB1", "NOTCH1", "MYC"]
sc.pl.umap(cancer, color=cancer_marker_genes_II, use_raw=True)

# Differentially expressed genes in each cluster 
sc.tl.rank_genes_groups(
    cancer, groupby="leiden", method="wilcoxon", key_added="dea_leiden"
)
# Get the result dictionary
result = cancer.uns['dea_leiden']

# 'names' contains the ranked gene names per cluster
groups = result['names'].dtype.names  # cluster names

for group in groups:
    print(f"Top 10 DE genes for cluster {group}:")
    top_genes = result['names'][group][:30]
    print(top_genes)
    print("\n")
sc.tl.dendrogram(cancer, groupby='leiden')
sc.pl.rank_genes_groups_dotplot(
    cancer, groupby="leiden", standard_scale="var", n_genes=20, key="dea_leiden"
)















