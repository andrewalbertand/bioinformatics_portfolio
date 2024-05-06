# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

## The data can be found at: https://www.cbioportal.org/study/summary?id=gbm_tcga ##
# Load data 
file_path = '/path/to/dataset/gbm_tcga_clinical_data.tsv'
tcga_df = pd.read_csv(file_path, sep = '\t')

# Conduct preliminary EDA
print(tcga_df.head())
print(len(tcga_df))
print(tcga_df.shape)
print(tcga_df.columns)

# Drop rows where 'Diagnosis Age' or 'Cancer Type Detailed' is missing
clean_df = tcga_df.dropna(subset=['Diagnosis Age', 'Cancer Type Detailed'])

# Create a boxplot to see the distribution of 'Diagnosis Age' across different cancer types
plt.figure(figsize=(12, 8))
sns.boxplot(x='Cancer Type Detailed', y='Diagnosis Age', data=clean_df)
plt.xticks(rotation=45)
plt.title('Distribution of Diagnosis Age by Cancer Type')
plt.xlabel('Cancer Type Detailed')
plt.ylabel('Diagnosis Age')
plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap
plt.show()

# Convert any categorical data that's incorrectly formatted as numeric
tcga_df['Tissue Source Site'] = tcga_df['Tissue Source Site'].astype('category')

# Fill missing values in 'TMB (nonsynonymous)' for visualization purposes
tcga_df['TMB (nonsynonymous)'] = tcga_df['TMB (nonsynonymous)'].fillna(tcga_df['TMB (nonsynonymous)'].median())

# Advanced visualizations
# Histogram of TMB
plt.figure(figsize=(10, 6))
sns.histplot(tcga_df['TMB (nonsynonymous)'], kde=True)
plt.title('Distribution of Tumor Mutational Burden (TMB)')
plt.xlabel('TMB (nonsynonymous)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of Diagnosis Age vs. TMB
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Diagnosis Age', y='TMB (nonsynonymous)', hue='Cancer Type Detailed', data=tcga_df)
plt.title('Diagnosis Age vs. Tumor Mutational Burden by Cancer Type')
plt.xlabel('Diagnosis Age')
plt.ylabel('TMB (nonsynonymous)')
plt.legend(title='Cancer Type Detailed')
plt.show()

# Correlation analysis
# Compute correlation matrix
correlation_matrix = tcga_df[['Diagnosis Age', 'TMB (nonsynonymous)', 'Disease Free (Months)']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Selected Features')
plt.show()

# Define age categories
bins = [0, 30, 50, 70, 100]
labels = ['0-30', '31-50', '51-70', '71+']
tcga_df['Age Group'] = pd.cut(tcga_df['Diagnosis Age'], bins=bins, labels=labels)

# Display the new dataframe with age groups
print(tcga_df[['Diagnosis Age', 'Age Group']].head())

# Standardize the data
features = ['Diagnosis Age', 'TMB (nonsynonymous)', 'Disease Free (Months)']
x = tcga_df[features].dropna()
x_scaled = StandardScaler().fit_transform(x)

# PCA transformation
pca = PCA(n_components=2)
components = pca.fit_transform(x_scaled)

# Create a new DataFrame for the PCA components
pca_df = pd.DataFrame(components, columns=['PCA1', 'PCA2'], index=x.index)

# Join the PCA components back to the original DataFrame
tcga_df = tcga_df.join(pca_df)

# Visualize the PCA results (only for rows where PCA was computed)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cancer Type Detailed', data=tcga_df.dropna(subset=['PCA1', 'PCA2']))
plt.title('PCA of Key Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()