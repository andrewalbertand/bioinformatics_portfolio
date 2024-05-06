# Import libraries
import os
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt

# Load data 
## The data used in this code can be found here: https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-2828/sdrf ##
file_path = '/path/to/file/chip-seq-data'

# List all BED files in the directory
bed_files = glob.glob(os.path.join(file_path, '*.bed'))

# Load and concatenate all BED files into a single DataFrame
df_list = []
for file in bed_files:
    df = pd.read_csv(file, sep='\t', header=None, names=['chrom', 'start', 'end', 'peak_name', 'score'])
    df_list.append(df)
all_peaks = pd.concat(df_list, ignore_index=True)

# Display the first few rows of the combined DataFrame
print(all_peaks.head())

# Basic statistics of peak scores
print("Basic Statistics of Peak Scores:")
print(all_peaks['score'].describe())

# Data visualization
sns.histplot(all_peaks['score'], bins=30, kde=True)
plt.title('Distribution of Peak Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()

# Distribution of peaks per chromosome
chrom_counts = all_peaks['chrom'].value_counts()
plt.figure(figsize=(10, 8))
sns.barplot(x=chrom_counts.index, y=chrom_counts.values, palette='viridis')
plt.title('Distribution of Peaks per Chromosome')
plt.xlabel('Chromosome')
plt.ylabel('Number of Peaks')
plt.xticks(rotation=45)
plt.show()

# Boxplot to show the distribution of peak scores per chromosome
plt.figure(figsize=(12, 8))
sns.boxplot(x='chrom', y='score', data=all_peaks, palette='coolwarm')
plt.title('Peak Score Distribution per Chromosome')
plt.xlabel('Chromosome')
plt.ylabel('Peak Score')
plt.xticks(rotation=45)
plt.show()

# Examining the width of peaks
all_peaks['width'] = all_peaks['end'] - all_peaks['start']
plt.figure(figsize=(8, 6))
sns.histplot(all_peaks['width'], bins=30, color='green', kde=True)
plt.title('Distribution of Peak Widths')
plt.xlabel('Width')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of Peak Score vs. Width to see if there's any correlation
plt.figure(figsize=(10, 6))
sns.scatterplot(x='width', y='score', data=all_peaks, alpha=0.6)
plt.title('Peak Score vs. Peak Width')
plt.xlabel('Width')
plt.ylabel('Score')
plt.show()