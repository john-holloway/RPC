#02/06/2025
### data analysis of metastatic LN

import scanpy as sc
import anndata as ad
import numpy as np
import bbknn
import pandas as pd
import glob
import os
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
import matplotlib.pyplot as plt
import seaborn as sns
import infercnvpy as cnv
import ktplotspy as kpy
from pathlib import Path




### load in h5ad objects
OAC_met = sc.read("/home/itrg/University/RPC/sc_data/OAC_mets.h5ad")

### Set parameters here 
npcs = 20
hvgs = 2500
res = 0.7
mindist = 0.3

### Normalisation  
# remove all NaNs
OAC_met.X = np.nan_to_num(OAC_met.X)
# save raw count data 
# Saving count data
OAC_met.layers["counts"] = OAC_met.X.copy()
# Normalizing to median total counts
sc.pp.normalize_total(OAC_met)
# Logarithmize the data
sc.pp.log1p(OAC_met)
# freeze the state in '.raw'
OAC_met.raw = OAC_met

### Feature selection 
sc.pp.highly_variable_genes(OAC_met, n_top_genes=hvgs, batch_key="sample")
sc.pl.highly_variable_genes(OAC_met)
# actually filtering 
OAC_met = OAC_met[:, OAC_met.var.highly_variable]
# to return to the anndata object prefiltering this is the rescue code
# adata.raw.to_adata()

### Dimensionalilty reduction (PCA)
sc.pp.scale(OAC_met)
sc.tl.pca(OAC_met)
sc.pl.pca_variance_ratio(OAC_met, n_pcs=npcs, log=True)
sc.pl.pca(
    OAC_met,
    color=["sample", "sample", "pct_counts_mt", "pct_counts_mt"],
    dimensions=[(0, 1), (2, 3), (0, 1), (2, 3)],
    ncols=2,
    size=20,
)

### BBKNN 
# batch correction 
# bbknn
sc.external.pp.bbknn(OAC_met, batch_key="sample", n_pcs=npcs)

### PCA plot post batch correction 
sc.pl.pca(OAC_met, color=["sample"], dimensions=[(0, 1)], ncols=2, size=20,)

### Investigatiing the PCA outlier
# Store the PCA coordinates in a new column in .obs
OAC_met.obs[['pca1', 'pca2']] = OAC_met.obsm['X_pca'][:, :2]
# Look at cells with extreme PC1 values
OAC_met.obs.sort_values(by='pca1').head()
OAC_met.obs.sort_values(by='pca1').tail()
### Identifying the outlier
outlier_cell = 'ACCCGTCTATGT'  # replace with real cell ID
# Inspect metadata
print(OAC_met.obs.loc[outlier_cell])
# Inspect raw counts or log-normalized values
print(OAC_met[outlier_cell].X)
# Optionally check QC metrics like:
# - n_genes_by_counts
# - total_counts
# - pct_counts_mt
print(OAC_met.obs.loc[outlier_cell, ['total_counts', 'n_genes_by_counts', 'pct_counts_mt']])

### remove the outlier
OAC_met = OAC_met[OAC_met.obs_names != outlier_cell].copy()
# replot PCA
sc.pl.pca(OAC_met, color=["sample"], dimensions=[(0, 1)], ncols=2, size=20,)
sc.pl.pca(OAC_met, color=["total_counts"], dimensions=[(0, 1)], ncols=2, size=20,)
sc.pl.pca(OAC_met, color=["n_genes_by_counts"], dimensions=[(0, 1)], ncols=2, size=20,)
sc.pl.pca(OAC_met, color=["pct_counts_mt"], dimensions=[(0, 1)], ncols=2, size=20,)

### Nearest neighbour graph
# sc.pp.neighbors(OAC_met) ### this is not required if running bbknn
sc.tl.umap(OAC_met, min_dist=mindist)
sc.pl.umap(
    OAC_met,
    color="sample",
    # Setting a smaller point size to get prevent overlap
    size=20,
)



### clustering
# Leiden graph-clustering method
# Using the igraph implementation and a fixed number of iterations can be significantly faster, especially for larger datasets
sc.tl.leiden(OAC_met, resolution=res)
sc.pl.umap(OAC_met, color=["leiden"], size=20) # change colour with palette parameter

### Labelled UMAP
cluster_to_celltype = {
    '0': 'T cells',
    '1': 'Cancer cells',
    '2': 'T cells',
    '3': 'B cells',
    '4': 'Cancer cells',
    '5': 'Cancer cells',
    '6': 'Cancer cells',
    '7': 'Myeloid cells',
    '8': 'Stromal cells',
    '9': 'T cells',
    '10': 'Endothelial cells',
    '11': 'Plasma cells',
    '12': 'Cancer cells',
    '13': 'Mast cells'
    # Add more as needed
}

# Create the new column
OAC_met.obs['cell_type'] = OAC_met.obs['leiden'].map(cluster_to_celltype)
sc.pl.umap(OAC_met, color=["cell_type"], size=20)

#### look at cell proportion differeces
# make treatment column
OAC_met.obs["treatment"] = OAC_met.obs["sample"].map({
    "OAC26_M": "Naive",
    "OAC35_M": "CRT"
})

# Count number of cells per treatment per cell type
counts = OAC_met.obs.groupby(["treatment", "cell_type"]).size().unstack(fill_value=0)

# Convert to proportions (row-wise)
proportions = counts.div(counts.sum(axis=1), axis=0)

print(proportions)

# Flatten proportions for plotting
prop_df = proportions.reset_index().melt(id_vars="treatment", var_name="cell_type", value_name="proportion")

plt.figure(figsize=(10, 6))
sns.barplot(data=prop_df, x="cell_type", y="proportion", hue="treatment")
plt.xticks(rotation=45)
plt.title("Cell type proportion by treatment")
plt.tight_layout()
plt.show()
print(OAC_met.obs["treatment"].value_counts())

### stacked barplot 
# run all together at once
proportions.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 6),
    colormap="tab20"  # or use your preferred colormap
)
plt.ylabel("Proportion")
plt.xlabel("Treatment")
plt.title("Stacked Cell Type Proportions by Treatment")
plt.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
##################################################################################################################################
####################################### Now look at TME for each treatment arm ##################################

### split into individual treatment objects 

# Subset where sample == "OAC26_M"
naive = OAC_met[OAC_met.obs["sample"] == "OAC26_M"].copy()

# Subset where sample == "OAC35_M"
CRT = OAC_met[OAC_met.obs["sample"] == "OAC35_M"].copy()

### plot UMAP for each patients LN

sc.pl.umap(naive, color=["cell_type"], size=20)
sc.pl.umap(CRT, color=["cell_type"], size=20)

### extract the count matices for corr plot
# Extract normalized expression matrix from adata.X
matrix_naive = naive.X  # This is cells x genes

# Convert to dense if it's a sparse matrix
if not isinstance(matrix_naive, np.ndarray):
    matrix_naive = matrix_naive.toarray()

# Create a DataFrame: genes x cells
df = pd.DataFrame(matrix_naive.T, index=naive.var_names, columns=naive.obs_names)

# Save to CSV (or TSV)
df.to_csv("/home/itrg/University/RPC/sc_analysis/Update/Aim 2/matrix_naive.csv")

# CRT
matrix_CRT = CRT.X
if not isinstance(matrix_CRT, np.ndarray):
    matrix_CRT = matrix_CRT.toarray()
df=pd.DataFrame(matrix_CRT.T, index=CRT.var_names, columns=CRT.obs_names)
df.to_csv("/home/itrg/University/RPC/sc_analysis/Update/Aim 2/matrix_CRT.csv")

# extracting the variable genes list 
# Step 1: Identify highly variable genes
# Step 2: Sort all genes by variability (descending)
hv_df = CRT.var.copy()
hv_df_sorted = hv_df.sort_values(by='dispersions_norm', ascending=False)

# Step 3: Get list of top variable gene names
top_variable_genes = hv_df_sorted.index.tolist()

# Convert list to a DataFrame
df_genes = pd.DataFrame(top_variable_genes, columns=["gene"])

# Save to CSV (or .txt)
df_genes.to_csv("/home/itrg/University/RPC/sc_analysis/Update/Aim 2/CRT_variable_genes.csv", index=False)

### cellphone db 
### naive 
from IPython.display import HTML, display
from cellphonedb.utils import db_releases_utils

display(HTML(db_releases_utils.get_remote_database_versions_html()['db_releases_html_table']))

# define version and path and then download db
# -- Version of the databse
cpdb_version = 'v5.0.0'

# -- Path where the input files to generate the database are located
cpdb_target_dir = os.path.join('/home/itrg/University/RPC/sc_analysis/OAC_mets/cellphonedb/', cpdb_version)
# download
from cellphonedb.utils import db_utils
db_utils.download_database(cpdb_target_dir, cpdb_version)

# make the file requirements 
print(naive.obs.head())
print(naive.obs.columns)

# make the file requirements 

meta_df = pd.DataFrame({
    "Cell": naive.obs_names,
    "cell_type": naive.obs["cell_type"]  # or whatever the column is named
})
meta_df.to_csv("/home/itrg/University/RPC/sc_analysis/Old/OAC_mets/cellphonedb/naive_meta.txt", sep="\t", index=False)

# set file paths 
cpdb_file_path = "/home/itrg/University/RPC/sc_analysis/Old/OAC_mets/cellphonedb/v5.0.0/cellphonedb.zip"
meta_file_path = "/home/itrg/University/RPC/sc_analysis/Old/OAC_mets/cellphonedb/naive_meta.txt"
out_file_path = "/home/itrg/University/RPC/sc_analysis/Old/OAC_mets/cellphonedb/naive_out"

# Run cellphone db on all the data 
from cellphonedb.src.core.methods import cpdb_analysis_method

cpdb_results = cpdb_statistical_analysis_method.call(
    cpdb_file_path =cpdb_file_path,           # Example: "cellphonedb/data/cellphonedb.zip"
    meta_file_path = meta_file_path,             # Contains: Cell\tcell_type
    counts_file_path = naive,        # AnnData file with expression matrix
    counts_data = "hgnc_symbol",                  # Gene symbols must match CPDB format
    output_path = out_file_path                      # Output folder path (e.g. "./cpdb_out/")
)

### CRT
# make the file requirements 
print(CRT.obs.head())
print(CRT.obs.columns)

# make the file requirements 

meta_df = pd.DataFrame({
    "Cell": CRT.obs_names,
    "cell_type": CRT.obs["cell_type"]  # or whatever the column is named
})
meta_df.to_csv("/home/itrg/University/RPC/sc_analysis/Old/OAC_mets/cellphonedb/CRT_meta.txt", sep="\t", index=False)

# set file paths 
cpdb_file_path = "/home/itrg/University/RPC/sc_analysis/Old/OAC_mets/cellphonedb/v5.0.0/cellphonedb.zip"
meta_file_path = "/home/itrg/University/RPC/sc_analysis/Old/OAC_mets/cellphonedb/CRT_meta.txt"
out_file_path = "/home/itrg/University/RPC/sc_analysis/Old/OAC_mets/cellphonedb/CRT_out"

# Run cellphone db on all the data 
from cellphonedb.src.core.methods import cpdb_analysis_method

cpdb_results = cpdb_statistical_analysis_method.call(
    cpdb_file_path =cpdb_file_path,           # Example: "cellphonedb/data/cellphonedb.zip"
    meta_file_path = meta_file_path,             # Contains: Cell\tcell_type
    counts_file_path = CRT,        # AnnData file with expression matrix
    counts_data = "hgnc_symbol",                  # Gene symbols must match CPDB format
    output_path = out_file_path                      # Output folder path (e.g. "./cpdb_out/")
)

##################################################################################################################################
####################################### Making cellphonedb plots ##################################

##### read in data from cellphone db output 

DATADIR = Path("/home/itrg/University/RPC/sc_analysis/Update/Aim 2/")

### CRT 

#  output from CellPhoneDB
means_crt = pd.read_csv(DATADIR / "CRT_out" / "statistical_analysis_means_07_03_2025_155001.txt", sep="\t")
pvals_crt = pd.read_csv(DATADIR / "CRT_out" / "statistical_analysis_pvalues_07_03_2025_155001.txt", sep="\t")
decon_crt = pd.read_csv(DATADIR / "CRT_out" / "statistical_analysis_deconvoluted_07_03_2025_155001.txt", sep="\t")

### NAIVE 

#  output from CellPhoneDB
means_naive = pd.read_csv(DATADIR / "naive_out" / "statistical_analysis_means_07_03_2025_154806.txt", sep="\t")
pvals_naive = pd.read_csv(DATADIR / "naive_out" / "statistical_analysis_pvalues_07_03_2025_154806.txt", sep="\t")
decon_naive = pd.read_csv(DATADIR / "naive_out" / "statistical_analysis_deconvoluted_07_03_2025_154806.txt", sep="\t")


### heat maps 
# crt
kpy.plot_cpdb_heatmap(pvals=pvals_crt, figsize=(5, 5), title="Sum of significant interactions CRT")
# naive
kpy.plot_cpdb_heatmap(pvals=pvals_naive, figsize=(5, 5), title="Sum of significant interactions naive")
# subset like this if required
# kpy.plot_cpdb_heatmap(pvals=pvals, cell_types=["NK cell", "pDC", "B cell", "CD8T cell"], figsize=(4, 4), title="Sum of significant interactions")

### dot plot 
# cell type 1 send signal and cell type 2 recieves signal
# could do tumour cells to all others 
# genes can be changed to whatever 
# gene family can be specified (genefamily = "chemokines")
kpy.plot_cpdb(
    adata=CRT,
    cell_type1="Cancer cells",
    cell_type2=".",  # this means all cell-types
    means=means_crt,
    pvals=pvals_crt,
    celltype_key="cell_type",
    figsize=(13, 4),
    title="interacting interactions!",
)

kpy.plot_cpdb(
    adata=naive,
    cell_type1="Cancer cells",
    cell_type2=".",  # this means all cell-types
    means=means_naive,
    pvals=pvals_naive,
    celltype_key="cell_type",
    figsize=(13, 4),
    title="interacting interactions!",
)

kpy.plot_cpdb_chord(
    adata=CRT,
    cell_type1="Cancer cells",
    cell_type2=".",
    means=means_crt,
    pvals=pvals_crt,
    deconvoluted=decon_crt,
    celltype_key="cell_type",
    link_kwargs={"direction": 1, "allow_twist": True, "r1": 95, "r2": 90},
    sector_text_kwargs={"color": "black", "size": 12, "r": 105, "adjust_rotation": True},
    legend_kwargs={"loc": "center", "bbox_to_anchor": (1, 1), "fontsize": 8},
    link_offset=1,
)

kpy.plot_cpdb_chord(
    adata=naive,
    cell_type1="Cancer cells",
    cell_type2=".",
    means=means_naive,
    pvals=pvals_naive,
    deconvoluted=decon_crt,
    celltype_key="cell_type",
    link_kwargs={"direction": 1, "allow_twist": True, "r1": 95, "r2": 90},
    sector_text_kwargs={"color": "black", "size": 12, "r": 105, "adjust_rotation": True},
    legend_kwargs={"loc": "center", "bbox_to_anchor": (1, 1), "fontsize": 8},
    link_offset=1,
)

##################################################################################################################################
####################################### Correlation plots ########################################################################


### make corr plot 
# make new anndata objects
CRT_corr = CRT.copy()
# subset to top 500 genes 
sc.pp.highly_variable_genes(CRT_corr, n_top_genes=hvgs, batch_key="sample")
# Extract expression matrix (cells x genes)
expr = CRT_corr.X.T  # transpose to genes x cells

# Compute Spearman correlation (genes x genes)
corr, _ = spearmanr(expr, axis=1)
corr_df = pd.DataFrame(corr, index=CRT_corr.var_names, columns=CRT_corr.var_names)

# Compute linkage
linkage = sch.linkage(1 - corr_df, method='average')  # 1 - corr to convert to distance
# Cut dendrogram to get clusters — you can choose number of clusters (e.g. 10)
clusters = sch.fcluster(linkage, t=10, criterion='maxclust')
# Map gene to cluster
gene_clusters = pd.Series(clusters, index=corr_df.index)

# Assuming corr_df is your gene-gene correlation DataFrame
sns.clustermap(
    corr_df,
    row_linkage=linkage,
    col_linkage=linkage,
    cmap='vlag',
    xticklabels=False,
    yticklabels=False,
    figsize=(10, 10)
)
plt.show()


###############

# Transpose expression to get genes x cells
expr = CRT_corr.X.T
genes = CRT_corr.var_names

# Ensure dense matrix
if hasattr(expr, "toarray"):
    expr = expr.toarray()

# Remove genes with zero variance
gene_var = np.var(expr, axis=1)
nonzero_idx = gene_var > 0
expr = expr[nonzero_idx]
genes = genes[nonzero_idx]

# Compute Spearman correlation (genes x genes)
corr_matrix, _ = spearmanr(expr, axis=1)
corr_df = pd.DataFrame(corr_matrix, index=genes, columns=genes)

# Convert to distance matrix
dist_matrix = 1 - corr_df

# Remove NaNs (caused by undefined Spearman values)
mask = ~np.isnan(dist_matrix).any(axis=0)
dist_matrix = dist_matrix.loc[mask, mask]

# Symmetrize and convert to condensed format
dist_matrix = (dist_matrix + dist_matrix.T) / 2
condensed = squareform(dist_matrix.values)

# Hierarchical clustering
link = linkage(condensed, method="average")

# Cut dendrogram into clusters
clusters = fcluster(link, t=10, criterion='maxclust')

# Map genes to clusters
gene_clusters = pd.Series(clusters, index=dist_matrix.index)































