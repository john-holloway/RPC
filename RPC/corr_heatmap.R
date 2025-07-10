# 10/07/2025
# correlation co expression clustering

## libraries
library(ComplexHeatmap)
library(circlize)
library(dendextend)
library(readr)


### read in data
# read in the gene gene corr matrix from python 
# read in a normalised gene x cell matrix that has been normalised 
data.genes <- read_csv("University/RPC/sc_analysis/Update/Aim 2/matrix_naive.csv")
# read in the variable genes list descening with most variable at top
variable_genes <- read_csv("University/RPC/sc_analysis/Update/Aim 2/naive_variable_genes.csv")
# take top 500 genes 
variable_genes <- variable_genes[1:250, ]

#Heatmaps for Top 50 cluster genes
genes <- unique(c(variable_genes$gene))
#genes <- genes[!genes=="PNMA5"]
# 1. Make sure gene names are row names
data.genes <- as.data.frame(data.genes)
rownames(data.genes) <- data.genes$GENE
data.genes$GENE <- NULL
#transpose/scale
data.genes <- as.matrix(t(data.genes[genes,]))
data.genes <- scale(data.genes,center = T,scale = T)

cormat<-signif(cor(data.genes,use = "pairwise.complete.obs",method = "pearson"),2)
rowSums(is.na(cormat))
cormat[is.na(cormat)] <- 0
dend <- as.dendrogram(hclust(as.dist(1-cormat),method = "ward.D2"))

plot(color_branches(dend, k=9),leaflab = "none")

col_scale = colorRamp2(c(-0.5, -0.1,0.05,0.4, 1), c("dodgerblue4","dodgerblue1","white", "orangered2","orangered4"))
png("University/RPC/sc_analysis/Update/Aim 2/cor.heatmap.dges.top50.cancer.png",width = 40,height = 40,res = 600,units = "cm")
h1 <- Heatmap(cormat, col = col_scale,show_column_names = FALSE,row_names_gp = gpar(fontsize = 1),name ="Correlation", column_title = "Top 50 Cluster Markers",cluster_rows = dend,cluster_columns = dend,row_split = 9,column_split = 9,row_title = NULL)
draw(h1)
dev.off()

clusters <- cutree(dend, k=9, order_clusters_as_data = FALSE,use_labels_not_values = F)
table(clusters)