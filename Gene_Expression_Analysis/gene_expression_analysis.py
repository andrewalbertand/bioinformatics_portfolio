# Import libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from google.colab import drive
import sys
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

## The data is downloadable from: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264477 ## 
# Load the gene expression data
file_path = '/path/to/dataset/GSE264477_Tumor.T.RPKM.csv.gz'
data = pd.read_csv(file_path, compression='gzip', index_col=0)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
principal_components = pca.fit_transform(data_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'], index=data.index)

# Visualize the PCA results
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', data=principal_df)
plt.title('PCA of Gene Expression Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
plt.grid(True)
plt.show()

# Explained variance ratio
print("Explained variance per principal component: {}".format(pca.explained_variance_ratio_))

# Correlation Heatmap of the data
correlation_matrix = pd.DataFrame(data).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap of Gene Expression Data')
plt.show()

# K-means Clustering on PCA results
kmeans = KMeans(n_clusters=3, random_state=0).fit(principal_components)
data['Cluster'] = kmeans.labels_

# Plot PCA with KMeans cluster assignments
plt.figure(figsize=(10, 8))
sns.scatterplot(x=principal_df['Principal Component 1'], y=principal_df['Principal Component 2'], hue=data['Cluster'], palette='viridis')
plt.title('PCA with KMeans Clustering')
plt.show()

# Increase recursion limit
sys.setrecursionlimit(10000)  # Increase the recursion limit of the OS

# If the dataset is too large, use a smaller subset for the dendrogram
subset_size = 100  # Adjust 
indices = np.random.choice(data_scaled.shape[0], subset_size, replace=False)
data_subset = data_scaled[indices]
labels_subset = data.index[indices].tolist()

# Perform hierarchical clustering on the subset
linked_subset = linkage(data_subset, 'ward')

# Plot the dendrogram
plt.figure(figsize=(10, 8))
dendrogram(linked_subset, labels=labels_subset, leaf_rotation=90, leaf_font_size=8)
plt.title('Hierarchical Clustering Dendrogram (Subset)')
plt.show()
