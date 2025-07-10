### 02/06/2025 
### preprocessing and qc of objects 
# libraries 
import pandas as pd
import scanpy as sc
import scrublet as scr
import numpy as np

### read in data and form into anndata objects
# create function 
def create_anndata(sample_dict):
    """
    Create AnnData objects from a dictionary of sample names and file paths.

    Parameters:
    - sample_dict (dict): keys are sample names, values are file paths to .dge.txt files

    Returns:
    - dict: keys are sample names, values are AnnData objects
    """
    adatas = {}

    for sample_name, file_path in sample_dict.items():
        print(f"Processing {sample_name}...")
        
        # Load the count matrix
        dge = pd.read_csv(file_path, sep="\t", index_col=0)
        dge_T = dge.T

        # Create AnnData object
        adata = sc.AnnData(X=dge_T)
        adata.obs_names = dge_T.index
        adata.var_names = dge_T.columns

        # Add sample name to metadata
        adata.obs["sample"] = sample_name

        # Store in dictionary
        adatas[sample_name] = adata

    return adatas

# set params
sample_paths = {
    "OAC26_M": "C:/Users/jh9u24/OneDrive - University of Southampton/University/MRes/RPC/OAC_sc_data/OAC_1600_cores/OAC26rLN/OAC26rLN.dge.txt",
    "OAC35_M": "C:/Users/jh9u24/OneDrive - University of Southampton/University/MRes/RPC/OAC_sc_data/OAC_1600_cores/OAC35/OAC35.dge.txt",
    "OAC20_U": "C:/Users/jh9u24/OneDrive - University of Southampton/University/MRes/RPC/OAC_sc_data/OAC_1600_cores/OAC20_2210/OAC20_2210.dge.txt",
    "OAC36_U": "C:/Users/jh9u24/OneDrive - University of Southampton/University/MRes/RPC/OAC_sc_data/OAC_1600_cores/OAC36/OAC36.dge.txt",
}

# run function 
anndata_dict = create_anndata(sample_paths)

# Unpack dictionary into individual AnnData objects
OAC26_M_adata = anndata_dict["OAC26_M"]
OAC35_M_adata = anndata_dict["OAC35_M"]
OAC20_U_adata = anndata_dict["OAC20_U"]
OAC36_U_adata = anndata_dict["OAC36_U"]

### QC 
def run_qc_on_list(*adatas):
    """
    Run QC on a list of AnnData objects and return them in the same order.

    Parameters:
    - *adatas: any number of AnnData objects

    Returns:
    - tuple: the same AnnData objects after QC
    """
    updated = []

    for i, adata in enumerate(adatas):
        print(f"Running QC on sample {i+1}...")

        # Annotate gene types
        adata.var["mt"] = adata.var_names.str.startswith("MT-")  # Mitochondrial
        adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))  # Ribosomal
        adata.var["hb"] = adata.var_names.str.contains(r"^HB(?!P)")  # Hemoglobin (excluding HBP)

        # Compute QC metrics
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
        )

        updated.append(adata)

    return tuple(updated)

OAC26_M_adata, OAC35_M_adata, OAC20_U_adata, OAC36_U_adata = run_qc_on_list(
    OAC26_M_adata, OAC35_M_adata, OAC20_U_adata, OAC36_U_adata
)

### visualising QC
# OAC26
sc.pl.violin(
    OAC26_M_adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
)

sc.pl.scatter(OAC26_M_adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt")

### visualising qc
# OAC35
sc.pl.violin(
    OAC35_M_adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
)

sc.pl.scatter(OAC35_M_adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt")

### visualising qc
# OAC36
sc.pl.violin(
    OAC36_U_adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
)

sc.pl.scatter(OAC36_U_adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt")

### visualising qc
# OAC20
sc.pl.violin(
    OAC20_U_adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
)

sc.pl.scatter(OAC20_U_adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt")

### filtering and doublet detection 
print(OAC26_M_adata.n_obs)
print(OAC35_M_adata.n_obs)
print(OAC36_U_adata.n_obs)
print(OAC20_U_adata.n_obs)
def filtering(*adatas):
    """
    Filter cells and genes, and run Scrublet for doublet detection on each AnnData object.

    Parameters:
    - *adatas: One or more AnnData objects

    Returns:
    - tuple: Filtered AnnData objects (same order)
    """
    filtered = []

    for i, adata in enumerate(adatas):
        print(f"\nFiltering sample {i+1}...")

        # Filter cells and genes
        sc.pp.filter_cells(adata, min_genes=100)
        sc.pp.filter_genes(adata, min_cells=3)
        # try 10 and then try 15 see what the difference is 
        adata = adata[adata.obs.pct_counts_mt < 10]
        
        print(f"Number of cells: {adata.n_obs}")

        # Run Scrublet
        try:
            sc.pp.scrublet(adata, batch_key="sample")
            print("Scrublet successful")
        except Exception as e:
            print(f"Scrublet failed for sample {i+1}: {e}")

        filtered.append(adata)
        
        print(f"Number of cells: {adata.n_obs}")

    return tuple(filtered)

OAC26_M_adata, OAC35_M_adata, OAC20_U_adata, OAC36_U_adata = filtering(
    OAC26_M_adata, OAC35_M_adata, OAC20_U_adata, OAC36_U_adata
)

### merge datasets 
# concat
OAC_mets = sc.concat([OAC26_M_adata, OAC35_M_adata], join='outer', label='batch', keys=['OAC26', 'OAC35'])
OAC_ua = sc.concat([OAC20_U_adata, OAC36_U_adata], join='outer', label='batch', keys=['OAC20', 'OAC36'])

### QC check again 
# mets
sc.pl.violin(
    OAC_mets,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
)

sc.pl.scatter(OAC_mets, "total_counts", "n_genes_by_counts", color="pct_counts_mt")

print('***************************************************')

# OAC20
sc.pl.violin(
    OAC_ua,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
)

sc.pl.scatter(OAC_ua, "total_counts", "n_genes_by_counts", color="pct_counts_mt")

# save the anndata obj:
OAC_mets.write("C:/Users/jh9u24/OneDrive - University of Southampton/University/MRes/RPC/OAC_sc_data/OAC_1600_cores/OAC_mets.h5ad")
OAC_ua.write("C:/Users/jh9u24/OneDrive - University of Southampton/University/MRes/RPC/OAC_sc_data/OAC_1600_cores/OAC_ua.h5ad")

