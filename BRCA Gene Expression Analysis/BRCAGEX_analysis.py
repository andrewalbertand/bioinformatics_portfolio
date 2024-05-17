# Import necessary libraries
import gseapy as gp
import requests
import pandas as pd
from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import mannwhitneyu
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE

## The dataset can be found at the following link: 
## https://portal.gdc.cancer.gov/v1/files/839a4656-f2ce-4460-858a-8e6b314695ef
file_path = '/path/to/dataset/augmented_gene_counts.tsv'

# Read in the datset (TSV)
gex_df = pd.read_csv(file_path, sep='\t')

# Filter and clean the data
# Remove rows where gene_id is null or starts with 'N_' as they are not informative
gex_df = gex_df[~gex_df['gene_id'].str.startswith('N_')]

# Drop rows with any missing values
gex_df = gex_df.dropna()

# Subsetting for demonstration purposes 
# Access to greater computational resources can expedite processing the whole dataset
gex_df_subset = gex_df.sample(n=1000, random_state=42)  # Subset 1000 samples for demonstration

# Identify and visualize the most highly expressed genes (top 20 by TPM)
top_genes = gex_df_subset.nlargest(20, 'tpm_unstranded')
plt.figure(figsize=(10, 8))
sns.barplot(x='tpm_unstranded', y='gene_name', data=top_genes)
plt.title('Top 20 Highly Expressed Genes by TPM')
plt.xlabel('TPM')
plt.ylabel('Gene Name')
plt.show()

# Remove rows with zero or near-zero values in unstranded and stranded_first columns
gex_df_filtered = gex_df_subset[(gex_df_subset['unstranded'] > 0) & (gex_df_subset['stranded_first'] > 0)]

# Calculate log2 fold change
gex_df_filtered['log2_fold_change'] = (gex_df_filtered['stranded_first'] + 1).apply(np.log2) - (gex_df_filtered['unstranded'] + 1).apply(np.log2)

# Perform Mann-Whitney U test between unstranded and stranded_first to calculate p-values
gex_df_filtered['p_value'] = gex_df_filtered.apply(lambda row: mannwhitneyu(
    [row['stranded_first']] * int(row['stranded_first'] + 1),
    [row['unstranded']] * int(row['unstranded'] + 1),
    alternative='two-sided'
).pvalue, axis=1)

# Basic volcano plot with p-values
plt.figure(figsize=(10, 8))
sns.scatterplot(data=gex_df_filtered, x='log2_fold_change', y=-np.log10(gex_df_filtered['p_value']), hue='gene_type')
plt.axhline(-np.log10(0.05), color='r', linestyle='--')
plt.axvline(-1, color='b', linestyle='--')
plt.axvline(1, color='b', linestyle='--')
plt.title('Volcano Plot of Differential Expression')
plt.xlabel('Log2 Fold Change')
plt.ylabel('-Log10 P-value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

# Clustering genes based on expression patterns
# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(gex_df_subset[['unstranded', 'stranded_first', 'stranded_second', 'tpm_unstranded', 'fpkm_unstranded', 'fpkm_uq_unstranded']])

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
pca_data = pca.fit_transform(scaled_data)

# Perform t-SNE on the PCA-reduced data
tsne = TSNE(n_components=2, random_state=42)
tsne_data = tsne.fit_transform(pca_data)

# Visualize t-SNE clustering
plt.figure(figsize=(10, 8))
sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], palette='viridis')
plt.title('t-SNE Clustering of Gene Expression')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

# KMeans clustering
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(pca_data)

# Visualize clustering
plt.figure(figsize=(10, 8))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters, palette='viridis')
plt.title('Gene Expression Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Add a column to indicate significance based on p-value and log2 fold change
significance_threshold = 0.05
log2fc_threshold = 1

gex_df_filtered['significant'] = (gex_df_filtered['p_value'] < significance_threshold) & (abs(gex_df_filtered['log2_fold_change']) > log2fc_threshold)

# Print the number of significant genes
num_significant_genes = gex_df_filtered['significant'].sum()
print(f'Number of significantly differentially expressed genes: {num_significant_genes}')

# Basic volcano plot with p-values and significant genes highlighted
plt.figure(figsize=(10, 8))
sns.scatterplot(data=gex_df_filtered, x='log2_fold_change', y=-np.log10(gex_df_filtered['p_value']), hue='significant', palette=['grey', 'red'])
plt.axhline(-np.log10(significance_threshold), color='r', linestyle='--')
plt.axvline(-log2fc_threshold, color='b', linestyle='--')
plt.axvline(log2fc_threshold, color='b', linestyle='--')
plt.title('Volcano Plot of Differential Expression')
plt.xlabel('Log2 Fold Change')
plt.ylabel('-Log10 P-value')
plt.legend(title='Significant', loc='upper right', bbox_to_anchor=(1.3, 1))
plt.show()

# Prepare the list of significant genes
upregulated_genes = gex_df_filtered[gex_df_filtered['log2_fold_change'] > log2fc_threshold]['gene_name'].dropna().tolist()
downregulated_genes = gex_df_filtered[gex_df_filtered['log2_fold_change'] < -log2fc_threshold]['gene_name'].dropna().tolist()

# Ensure the gene lists are not empty
if not upregulated_genes:
    print("No upregulated genes found.")
else:
    # Run GSEA for upregulated genes
    try:
        up_gsea = gp.enrichr(gene_list=upregulated_genes,
                             gene_sets=['KEGG_2016', 'GO_Biological_Process_2021', 'Reactome_2022'],
                             organism='human',
                             cutoff=0.1)  # Lower cutoff threshold
    except ValueError as e:
        print(f"GSEA for upregulated genes failed: {e}")

if not downregulated_genes:
    print("No downregulated genes found.")
else:
    # Run GSEA for downregulated genes
    try:
        down_gsea = gp.enrichr(gene_list=downregulated_genes,
                               gene_sets=['KEGG_2016', 'GO_Biological_Process_2021', 'Reactome_2022'],
                               organism='human',
                               cutoff=0.1)  # Lower cutoff threshold
    except ValueError as e:
        print(f"GSEA for downregulated genes failed: {e}")