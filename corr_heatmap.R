# 10/07/2025
# correlation co expression clustering

## libraries
library(ComplexHeatmap)
library(circlize)
library(dendextend)
library(readr)
library(purrr)
library(ggplot2)
library(cluster)
library(factoextra)
library(NbClust)

##

### read in data
# read in the gene gene corr matrix from python 
# read in a normalised gene x cell matrix that has been normalised 
data.genes <- read_csv("/home/itrg/University/RPC/sc_analysis/Update/Aim 2/matrix_CRT.csv")
# read in the variable genes list descening with most variable at top
variable_genes <- read_csv("/home/itrg/University/RPC/sc_analysis/Update/Aim 2/CRT_variable_genes.csv")
# take top 500 genes 
variable_genes <- variable_genes[1:70, ]

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
####################################################################################################################################
######################################### determing k means ########################################################################
# Elbow method
fviz_nbclust(cormat, kmeans, method = "wss") +
  geom_vline(xintercept = 2, linetype = 2)+
  labs(subtitle = "Elbow method")
# Silhouette method
fviz_nbclust(cormat, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")
# Gap statistic
# nboot = 50 to keep the function speedy. 
# recommended value: nboot= 500 for your analysis.
# Use verbose = FALSE to hide computing progression.
set.seed(123)
fviz_nbclust(cormat, kmeans, nstart = 25,  method = "gap_stat", nboot = 50)+
  labs(subtitle = "Gap statistic method")

# change k manually 
plot(color_branches(dend, k=9),leaflab = "none")

col_scale = colorRamp2(c(-0.5, -0.1,0.05,0.4, 1), c("dodgerblue4","dodgerblue1","white", "orangered2","orangered4"))
#png("University/RPC/sc_analysis/Update/Aim 2/cor.heatmap.dges.top50.cancer.png",width = 40,height = 40,res = 600,units = "cm")
h1 <- Heatmap(cormat, col = col_scale,show_column_names = FALSE,
              row_names_gp = gpar(fontsize = 5),name ="Correlation", 
              column_title = "Top 70 Variable Genes - CRT",cluster_rows = dend,
              cluster_columns = dend,
              row_split = 4,
              column_split = 4,
              row_title = NULL)
draw(h1)

#dev.off()

clusters <- cutree(dend, k=4, order_clusters_as_data = FALSE,use_labels_not_values = F)
table(clusters)

module1 <- names(clusters)[clusters==1]
module2 <- names(clusters)[clusters==2]
module3 <- names(clusters)[clusters==3]
module4 <- names(clusters)[clusters==4]
#module5 <- names(clusters)[clusters==5]
#module6 <- names(clusters)[clusters==6]
#module7 <- names(clusters)[clusters==7]
#module8 <- names(clusters)[clusters==8]
#module9 <- names(clusters)[clusters==9]












