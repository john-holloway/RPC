# 09/07/2025
### comparing between CRT and NAIVE for specific cell types 


import anndata as ad
import scanpy as sc

# read in data
OAC_met_mod = sc.read("/home/itrg/University/RPC/sc_data/OAC_mets_mod.h5ad")
OAC_pt_mod = sc.read("/home/itrg/University/RPC/sc_data/OAC_pt_mod.h5ad")

# merge to single anndata object
OAC_pt_mod = OAC_pt_mod[OAC_pt_mod.obs["patient"]=="OAC35"]
OAC_met_mod = OAC_met_mod[OAC_met_mod.obs["sample"]=="OAC35_M"]

OAC_CRT = ad.concat([OAC_pt_mod, OAC_met_mod])


### cell proportion 
# make treatment column
OAC_CRT.obs["location"] = OAC_CRT.obs["sample"].map({
    "OAC35TJ": "Primary Tumour",
    "OAC35_M": "Lymph Node",
    "OAC35TL": "Primary Tumour"
})

print(OAC_CRT.obs.columns)


# Count number of cells per treatment per cell type
counts = OAC_CRT.obs.groupby(["location", "cell_type"]).size().unstack(fill_value=0)
# Filter for only specific cell types in the count table
cell_types_of_interest = ["B cells", "T cells", "Myeloid cells", "NK cells", "Plasma cells", "Mast cells"]  # change as needed
counts = counts.loc[:, counts.columns.isin(cell_types_of_interest)]

print(counts.head(10))
# Convert to proportions (row-wise)
proportions = counts.div(counts.sum(axis=1), axis=0)

print(proportions)

# Flatten proportions for plotting
prop_df = proportions.reset_index().melt(id_vars="location", var_name="cell_type", value_name="proportion")

plt.figure(figsize=(10, 6))
sns.barplot(data=prop_df, x="cell_type", y="proportion", hue="location")
plt.xticks(rotation=45)
plt.title("Cell type proportion by Location")
plt.tight_layout()
plt.show()
print(OAC_CRT.obs["location"].value_counts())

### stacked barplot 
# run all together at once
proportions.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 6),
    colormap="tab20"  # or use your preferred colormap
)
plt.ylabel("Proportion")
plt.xlabel("Location")
plt.title("Stacked Cell Type Proportions by Treatment")
plt.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.xticks(rotation=0)
plt.show()


# volcano plot between cell types 
# 'Plasma cells', 'Smooth Muscle cells', 'Cancer cells', 'T cells', 'B cells, 
# 'Myeloid cells', 'Endothelial cells', 'NK cells', 'Mast cells', 'Stromal cells'
OAC_CRT.obs["cell_type"] = OAC_CRT.obs["cell_type"].replace("Fibroblasts", "Stromal cells")

cell_type_of_interest = "Stromal cells"
test_cells = OAC_CRT[OAC_CRT.obs['cell_type'] == cell_type_of_interest].copy()

sc.tl.rank_genes_groups(
    test_cells,
    groupby='location',
    groups=['Lymph Node'],
    reference='Primary Tumour',
    method='wilcoxon'  # or 't-test', 'logreg' depending on preference
)

df = sc.get.rank_genes_groups_df(test_cells, group='Lymph Node')
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

sig_mask = (df['pvals_adj'] < 0.05) & (df['logfoldchanges'].abs() > 1)
# Add gene labels for significant genes
for i, row in df[sig_mask].iterrows():
    plt.text(row['logfoldchanges'], row['-log10(padj)'], row['names'], 
             fontsize=8, ha='right', va='bottom')


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

