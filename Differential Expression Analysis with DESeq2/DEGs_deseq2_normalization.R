# GSE229344 
# Load necessary library
library(DESeq2)
library(ggplot2)
library(pheatmap)

# Set the file path
file_path <- '/path/to/dataset/GSE229344_raw_counts_GRCh38.p13_NCBI.tsv'

# Read in the dataset
countData <- as.matrix(read.table(file_path, header = TRUE, row.names = 1, sep = '\t', check.names = FALSE))

# View the first few rows
head(countData)

# Define sample conditions
condition <- factor(c(rep("PBS", 3), rep("Adenoviral_EV", 3), rep("Adenoviral_CYB5R3", 3)))

# Create the DataFrame for DESeq2
colData <- DataFrame(condition = condition)
rownames(colData) <- colnames(countData)

# Check the colData
print(colData)

# Create a DESeqDataSet
dds <- DESeqDataSetFromMatrix(countData = countData,
                              colData = colData,
                              design = ~ condition)

# Run the differential expression analysis
dds <- DESeq(dds)

# Get results for a specific comparison, e.g., Adenoviral_CYB5R3 vs PBS
results <- results(dds, contrast = c("condition", "Adenoviral_CYB5R3", "PBS"))

# Order results by p-value
resultsOrdered <- results[order(results$pvalue),]

# View the top differentially expressed genes
head(resultsOrdered)

# Results visualization
# MA Plot
plotMA(results, main = "MA-plot", ylim = c(-2, 2))

# Volcano Plot
volcano_data <- as.data.frame(results)
volcano_data$log2FoldChange <- as.numeric(volcano_data$log2FoldChange)
volcano_data$pvalue <- as.numeric(volcano_data$pvalue)
ggplot(volcano_data, aes(x=log2FoldChange, y=-log10(pvalue))) +
  geom_point(aes(color=pvalue < 0.05), alpha=0.4) +
  scale_color_manual(values = c("black", "red")) +
  theme_minimal() +
  labs(title = "Volcano Plot", x = "Log2 Fold Change", y = "-Log10 p-value")

# Principal Component Analysis (PCA)
rld <- rlog(dds, blind=FALSE)
plotPCA(rld, intgroup=c("condition"), returnData=FALSE)

# Subset data for heatmap to top 100 differentially expressed genes
top_genes <- head(order(results$pvalue), 100)
sub_countData <- assay(rld)[top_genes,]

# Heatmap of top differentially expressed genes
pheatmap(sub_countData, cluster_rows = TRUE, cluster_cols = TRUE, 
         main = "Top 100 Differentially Expressed Genes Heatmap",
         show_rownames = FALSE)