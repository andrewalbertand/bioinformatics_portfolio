Project Title: Differential Gene Expression Analysis in Response to Adenoviral Treatments
Overview
This project utilizes RNA-seq data to analyze the differential expression of genes in response to different adenoviral treatments, focusing on a study with the accession number GSE229344. The project employs statistical and bioinformatics tools to identify genes that show significant changes in expression, which could be crucial for understanding the molecular mechanisms of adenoviral impact.

Objectives
To identify differentially expressed genes between control (PBS) and adenoviral treatments (Adenoviral_EV, Adenoviral_CYB5R3).
To visualize data through MA plots, volcano plots, and heatmaps to understand expression patterns and the significance of changes.
Methods and Technologies
Data Preprocessing: Utilizing R to handle RNA-seq count data, prepare it for analysis, and ensure it's properly formatted for differential expression analysis.
Differential Expression Analysis: Using DESeq2 for determining differentially expressed genes across different conditions.
Visualization:
MA Plot: Showcasing magnitude of expression changes against mean average expression.
Volcano Plot: Highlighting statistically significant changes in gene expression.
Heatmap: Visualizing patterns of top differentially expressed genes.
Principal Component Analysis (PCA): Assessing sample variation and the effect of treatments on gene expression.
Results
Gene Expression Insights: Identification of top differentially expressed genes which may be key players in response to adenoviral treatments.
Visualization Outcomes: Effective use of plots to represent complex data, enabling easier interpretation of results.
Conclusions
This project highlights the application of bioinformatics in analyzing gene expression data, demonstrating the power of tools like DESeq2 and pheatmap in uncovering significant biological insights. The analysis not only aids in understanding the genomic effects of adenoviral treatments but also in identifying potential therapeutic targets.

Technologies Used
R
Bioconductor (DESeq2)
ggplot2
pheatmap
