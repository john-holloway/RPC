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
import gseapy as gp
from scipy.stats import spearmanr
import scipy.cluster.hierarchy as sch


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

# set marker genes
# cell types: Cancer cells, B cell, NK cell, T cell, monocyte, dendritic cell, lymphoid cell, macrophage, mast cell, non immune cell, adipocyte, fibroblast, pericyte, smooth muscle cell,
# non immune cell types: adipocyte, fibroblast reticular cell, CAF, pericyte, sooth muscle cell, endothelial cell, CXC12 reticular cells, capillary resident progenitor, follicular dendritic cells, marginal reticular cell, plasma cell, perivascular cell, subscapular sinus
marker_genes = {"B cell": ["CD79A", "MS4A1", "CD19"], "NK cell": ["NCAM1", "NKG7"], "T cell": ["CD3D", "CD3E"], "Monocyte": ["FCN1", "CD14"], "Dendritic cell": ["CD1C", "FCER1A", "CLEC10A"],
"Macrophage": ["C1QA", "CD68", "MS4A7"], "Mast cell": ["TPSAB1", "TPSB2", "CPA3"], "Cancer Associated Fibroblast": ["FAP", "ACTA2", "PDPN"], "Endothelial cell": ["PECAM1", "VWF"],
"Fibroblast": ["COL1A1", "DCN", "PDGFRA"], "Pericyte": ["PDGFRB", "RGS5"], "Smooth muscle cells": ["ACTA2", "MYH11", "TAGLN"], "Fibroblast reticular cell": ["PDPN", "CCL21"],
"Follicular dendritic cell": ["CR2", "FCER2", "CR1"], "Plasma cell": ["SDC1", "CD38", "MZB1"], "Cancer cell": ["EPCAM", "TGM2"]}
# subset to only the markers found in the data
marker_genes_in_data = {}
for ct, markers in marker_genes.items():
    markers_found = []
    for marker in markers:
        if marker in OAC_met.var.index:
            markers_found.append(marker)
    marker_genes_in_data[ct] = markers_found


marker_genes_II = ["MS4A1", "GNLY", "CD3D", "FCN1", "CD1C", "C1QA", "TPSAB1", "FAP", "PECAM1", "COL1A1", "PDGFRB", "ACTA2", "PDPN", "CR2", "SDC1", "EPCAM", "TGM2"]
sc.pl.umap(OAC_met, color=marker_genes_II, use_raw=True)

# Dot plot table
sc.pl.dotplot(
    OAC_met,
    groupby="leiden",
    var_names=marker_genes,
    standard_scale="var",  # standard scale: normalize each gene to range from 0 to 1
    use_raw=True
)

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

# Vionlin plot
# showing the same as above
sc.pl.violin(OAC_met, marker_genes_II, groupby='leiden')

# Differentially expressed genes in each cluster
sc.tl.rank_genes_groups(
    OAC_met, groupby="leiden", method="wilcoxon", key_added="dea_leiden"
)
# Get the result dictionary
result = OAC_met.uns['dea_leiden']

# 'names' contains the ranked gene names per cluster
groups = result['names'].dtype.names  # cluster names

for group in groups:
    print(f"Top 10 DE genes for cluster {group}:")
    top_genes = result['names'][group][:30]
    print(top_genes)
    print("\n")
sc.pl.rank_genes_groups_dotplot(
    OAC_met, groupby="leiden", standard_scale="var", n_genes=10, key="dea_leiden"
)
########## InferCNV
## parse GTF
gtf_file = "/home/itrg/University/RPC/sc_data/gencode.v48.annotation.gtf"# Path to your GTF file
gtf = pd.read_csv(
    gtf_file,
    sep='\t',
    comment='#',
    header=None,
    names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
)# Read GTF file, skipping comment lines
genes = gtf[gtf["feature"] == "gene"]# Filter for gene entries
# Extract gene_name from the 'attribute' column using string matching
def get_gene_name(attr):
    for item in attr.split(";"):
        item = item.strip()
        if item.startswith("gene_name"):
            return item.split('"')[1]  # get the value inside quotes
genes["gene_name"] = genes["attribute"].apply(get_gene_name)
# Build final DataFrame
gene_pos = genes[["gene_name", "seqname", "start", "end"]].copy()
gene_pos = gene_pos.rename(columns={"seqname": "chromosome"})
#gene_pos["chromosome"] = gene_pos["chromosome"].str.replace("^chr", "", regex=True)# Optional: remove "chr" prefix to match your adata.var
gene_pos = gene_pos.drop_duplicates(subset="gene_name")# Drop any duplicates

## merge with cancer.var
OAC_met.var = OAC_met.var.reset_index().rename(columns={"GENE": "gene_name"})  # 
OAC_met.var = OAC_met.var.merge(gene_pos, on="gene_name", how="left")# Merge
OAC_met.var = OAC_met.var.set_index("gene_name")# Set index back to gene names

## check
print(OAC_met.var[['chromosome', 'start', 'end']].head())

# Step 1: Get full gene annotation from raw.var
raw_var = OAC_met.raw.var.reset_index().rename(columns={"GENE": "gene_name"})

# Step 2: Merge genomic coordinates with full gene list
raw_var = raw_var.merge(gene_pos, on="gene_name", how="left").set_index("gene_name")

# Step 3: Build new AnnData object with correct dimensions and annotations
raw_adata = sc.AnnData(
    X=OAC_met.raw.X.copy(),
    obs=OAC_met.obs.copy(),
    var=raw_var
)


cnv.tl.infercnv(
    raw_adata,
    reference_key="cell_type",
    reference_cat=[
        "Cancer cells",
        'T cells', 
        'B cells', 
        'Stromal cells',
        'Endothelial cells', 
        'Plasma cells',
        'Mast cells'
    ],
    window_size=250,
)

cnv.pl.chromosome_heatmap(raw_adata, groupby="cell_type")

print(OAC_met.var.columns)  # After reset_index and rename
print(gene_pos.columns)     # The GTF-derived gene position DataFrame
print(OAC_met.var.reset_index().columns)

### make corr plot 
# Extract expression matrix (cells x genes)
expr = OAC_met.X.T  # transpose to genes x cells

# Compute Spearman correlation (genes x genes)
corr, _ = spearmanr(expr, axis=1)
corr_df = pd.DataFrame(corr, index=OAC_met.var_names, columns=OAC_met.var_names)

# Compute linkage
linkage = sch.linkage(1 - corr_df, method='average')  # 1 - corr to convert to distance
# Cut dendrogram to get clusters — you can choose number of clusters (e.g. 10)
clusters = sch.fcluster(linkage, t=10, criterion='maxclust')
# Map gene to cluster
gene_clusters = pd.Series(clusters, index=corr_df.index)

# Assuming corr_df is your gene-gene correlation DataFrame
sns.clustermap(
    corr_df,
    method='average',        # linkage method
    metric='correlation',    # distance metric
    cmap='vlag',             # diverging colormap for correlations
    figsize=(10, 10),
    yticklabels=False,
    xticklabels=False
)
plt.show()

###############################################################################################################################

### subset and recluster the cancer cells
# Create a Boolean mask for cancer clusters
cancer_mask = OAC_met.obs['leiden'].isin(['1', '4', '5', '6', '12'])

# Use the mask to subset the AnnData object (this is key!)
cancer = OAC_met[cancer_mask].copy()

# Now you can run Scanpy functions on this new AnnData object
sc.pp.neighbors(cancer)
sc.tl.leiden(cancer, resolution=0.5)
sc.tl.umap(cancer, min_dist=0.2)
sc.pl.umap(cancer, color=['sample'])


### Cancer Subset cell annotation
EMT_marker_genes = {"Epithelial cell": ["EPCAM", "CDH1"], "EMT TF": ["SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1", "TWIST2"], "Mesenchymal cell": ["VIM", "FN1", "CDH2", "MMP1", "SMN1"], "EMT-Inducing factors": ["TGFB1", "CTNNB1", "NOTCH1", "MYC"] }

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
cancer_activity_markers = ["MKI67", "MCM2", "PCNA"]
sc.pl.umap(cancer, color=cancer_activity_markers, use_raw=True)
sc.pl.umap(cancer, color=["EPCAM", "CDH1"], use_raw=True)

cluster_to_celltype_cancer = {
    '0': 'CRT cells',
    '1': 'Naive cells',
    '2': 'CRT cells',
    '3': 'CRT cells',
    '4': 'Naive cells',
    '5': 'CRT cells',
    '6': 'Naive cells'
}

# Create the new column
cancer.obs['cell_type'] = cancer.obs['leiden'].map(cluster_to_celltype_cancer)
sc.pl.umap(cancer, color=["cell_type"])


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

#####

### diff gene expression between mesenchymal and epithelial
sc.tl.rank_genes_groups(
    cancer,
    groupby='cell_type',  # <-- your custom column
    groups=['CRT cells'],
    reference='Naive cells',
    method='wilcoxon'
)

df = sc.get.rank_genes_groups_df(cancer, group='CRT cells')
# Save to CSV
df.to_csv('/home/itrg/University/RPC/sc_analysis/OAC_mets/dge_CRT_vs_naive.csv', index=False)

# Filter for upregulated genes
up_genes_df = df[(df['logfoldchanges'] > 1) & (df['pvals_adj'] < 0.05)]
up_genes = up_genes_df['names'].tolist()

# Filter for downregulated genes
down_genes_df = df[(df['logfoldchanges'] < -1) & (df['pvals_adj'] < 0.05)]
down_genes = down_genes_df['names'].tolist()

### volcano plot
# Get DE results for CD4_T vs CD8_T

# Calculate -log10 adjusted p-values
df['-log10(padj)'] = -np.log10(df['pvals_adj'])

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='logfoldchanges',
    y='-log10(padj)',
    data=df,
    hue=(df['pvals_adj'] < 0.05) & (df['logfoldchanges'].abs() > 1),
    palette={True: 'red', False: 'grey'},
    legend=False
)

# Add axis labels and title
plt.xlabel('Log2 Fold Change')
plt.ylabel('-Log10 Adjusted p-value')
plt.title('Volcano Plot')

# Optional: Add threshold lines
plt.axhline(-np.log10(0.05), color='blue', linestyle='--', linewidth=1)
plt.axvline(-1, color='black', linestyle='--', linewidth=1)
plt.axvline(1, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()

### kegg plot 

# Run KEGG enrichment analysis
enr = gp.enrichr(
    gene_list=down_genes,
    gene_sets='ChEA_2022',
    organism='Human',  # or 'Mouse'
    outdir='KEGG_results',  # folder to save results
    cutoff=0.05  # p-value threshold
)
# Barplot
gp.barplot(enr.res2d, title='KEGG Pathway Enrichment', cutoff=0.05, figsize=(6, 6))

### inferCNV on the cancer group
# We provide all immune cell types as "normal cells".
### gtf required for inferCNV

## parse GTF
gtf_file = "/home/itrg/University/RPC/sc_data/gencode.v48.annotation.gtf"# Path to your GTF file
gtf = pd.read_csv(
    gtf_file,
    sep='\t',
    comment='#',
    header=None,
    names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
)# Read GTF file, skipping comment lines
genes = gtf[gtf["feature"] == "gene"]# Filter for gene entries
# Extract gene_name from the 'attribute' column using string matching
def get_gene_name(attr):
    for item in attr.split(";"):
        item = item.strip()
        if item.startswith("gene_name"):
            return item.split('"')[1]  # get the value inside quotes
genes["gene_name"] = genes["attribute"].apply(get_gene_name)
# Build final DataFrame
gene_pos = genes[["gene_name", "seqname", "start", "end"]].copy()
gene_pos = gene_pos.rename(columns={"seqname": "chromosome"})
#gene_pos["chromosome"] = gene_pos["chromosome"].str.replace("^chr", "", regex=True)# Optional: remove "chr" prefix to match your adata.var
gene_pos = gene_pos.drop_duplicates(subset="gene_name")# Drop any duplicates

## merge with cancer.var
cancer.var = cancer.var.rename(columns={"GENE": "gene_name"})# Reset index to access gene names as a column
cancer.var = cancer.var.merge(gene_pos, on="gene_name", how="left")# Merge
cancer.var = cancer.var.set_index("gene_name")# Set index back to gene names

## check
print(cancer.var[['chromosome', 'start', 'end']].head())

# Step 1: Get full gene annotation from raw.var
raw_var = cancer.raw.var.reset_index().rename(columns={"GENE": "gene_name"})

# Step 2: Merge genomic coordinates with full gene list
raw_var = raw_var.merge(gene_pos, on="gene_name", how="left").set_index("gene_name")

# Step 3: Build new AnnData object with correct dimensions and annotations
raw_adata = sc.AnnData(
    X=cancer.raw.X.copy(),
    obs=cancer.obs.copy(),
    var=raw_var)


# make treatment column
cancer.obs["treatment"] = cancer.obs["sample"].map({
    "OAC26_M": "Naive",
    "OAC35_M": "CRT"
})

cnv.tl.infercnv(
    raw_adata,
    reference_key="treatment",
    reference_cat=[
        "Naive",
        "CRT",
    ],
    window_size=250,
)

cnv.pl.chromosome_heatmap(cancer, groupby="treatment")


###############################################################################################################################


### Cellphone Db
# download database from source
# display database versions 
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
print(OAC_met.obs.head())
print(OAC_met.obs.columns)

# make the file requirements 

meta_df = pd.DataFrame({
    "Cell": OAC_met.obs_names,
    "cell_type": OAC_met.obs["cell_type"]  # or whatever the column is named
})
meta_df.to_csv("/home/itrg/University/RPC/sc_analysis/OAC_mets/cellphonedb/OAC_met_meta.txt", sep="\t", index=False)

# set file paths 
cpdb_file_path = "/home/itrg/University/RPC/sc_analysis/OAC_mets/cellphonedb/v5.0.0/cellphonedb.zip"
meta_file_path = "/home/itrg/University/RPC/sc_analysis/OAC_mets/cellphonedb/OAC_met_meta.txt"
out_file_path = "/home/itrg/University/RPC/sc_analysis/OAC_mets/cellphonedb/out"

# Run cellphone db on all the data 
from cellphonedb.src.core.methods import cpdb_analysis_method

cpdb_results = cpdb_statistical_analysis_method.call(
    cpdb_file_path =cpdb_file_path,           # Example: "cellphonedb/data/cellphonedb.zip"
    meta_file_path = meta_file_path,             # Contains: Cell\tcell_type
    counts_file_path = OAC_met,        # AnnData file with expression matrix
    counts_data = "hgnc_symbol",                  # Gene symbols must match CPDB format
    output_path = out_file_path                      # Output folder path (e.g. "./cpdb_out/")
)



OAC_met.write("/home/itrg/University/RPC/sc_data/OAC_mets_mod.h5ad")


















































