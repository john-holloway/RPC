### Plotting cellphone db output
### from the metastatic LN 

import scanpy as sc
import anndata as ad
import pandas as pd
import ktplotspy as kpy
import matplotlib.pyplot as plt

from pathlib import Path

# read in the files
# 1) .h5ad file used for performing CellPhoneDB
DATADIR = Path("/home/itrg/University/RPC/sc_analysis/OAC_mets/cellphonedb/")

OAC_met = sc.read("/home/itrg/University/RPC/sc_data/OAC_mets.h5ad")

# 2) output from CellPhoneDB
means = pd.read_csv(DATADIR / "out" / "statistical_analysis_means_06_25_2025_102432.txt", sep="\t")
pvals = pd.read_csv(DATADIR / "out" / "statistical_analysis_pvalues_06_25_2025_102432.txt", sep="\t")
decon = pd.read_csv(DATADIR / "out" / "statistical_analysis_deconvoluted_06_25_2025_102432.txt", sep="\t")

# 3) plotting
# the original heatmap
kpy.plot_cpdb_heatmap(pvals=pvals, figsize=(5, 5), title="Sum of significant interactions")

