Project Title: TCGA Breast Cancer (BRCA) Gene Expression Analysis


Overview
This project utilizes RNA-Seq gene expression data from the TCGA BRCA study to analyze and visualize gene expression levels, identify differentially expressed genes, and cluster genes based on their expression patterns. By employing advanced data science and bioinformatics techniques, this study aims to uncover significant patterns and insights that could enhance our understanding of breast cancer biology and treatment responses.


Objectives
        1.        To clean and preprocess the RNA-Seq gene expression data from the TCGA BRCA project.
        2.        To visualize the distribution of gene expression levels across different conditions.
        3.        To identify and visualize the most highly expressed genes.
        4.        To perform differential expression analysis between different conditions.
        5.        To cluster genes based on their expression patterns using PCA and KMeans clustering.
        6.        To perform t-SNE for visualization of gene expression clusters.
        7.        To conduct Gene Set Enrichment Analysis (GSEA) on upregulated and downregulated genes.


Methods and Technologies


Data Preprocessing
        •        Data Cleaning: Removing non-informative rows (e.g., unmapped or multimapping reads) and handling missing values to ensure data quality.
        •        Gene Expression Quantification: Utilizing TPM (Transcripts Per Million), FPKM (Fragments Per Kilobase Million), and FPKM-UQ (Upper Quartile Normalized FPKM) for expression level comparisons.


Visualization
        •        Barplots: Visualizing the top 20 highly expressed genes by TPM.
        •        Histograms: Examining the distribution of TPM, FPKM, and FPKM-UQ values.
        •        Volcano Plot: Visualizing differential expression results with log2 fold change and p-values.


Statistical Analysis
        •        Differential Expression Analysis: Calculating log2 fold changes and performing the Mann-Whitney U test to identify differentially expressed genes between conditions.


Clustering
        •        Principal Component Analysis (PCA): Reducing dimensionality of the gene expression data to identify patterns.
        •        KMeans Clustering: Grouping genes based on expression patterns identified through PCA.
        •        t-SNE: Performing t-SNE for better visualization of gene expression clusters.
        •        Hierarchical Clustering: Creating dendrograms to visualize the hierarchical clustering of genes.


Gene Set Enrichment Analysis (GSEA)
        •        GSEA: Conducting GSEA on upregulated and downregulated genes to identify significantly enriched pathways and gene sets.


Results
        •        Top 20 Genes Visualization: Barplot showing the most highly expressed genes by TPM, providing insights into key genes involved in breast cancer.
        •        Volcano Plot: Clear visualization of differentially expressed genes, highlighting those with significant changes in expression between conditions.
        •        Clustering Visualization: Scatter plot of PCA components with KMeans clustering and t-SNE clustering, revealing distinct gene expression clusters.
        •        Hierarchical Clustering: Dendrogram illustrating the hierarchical clustering of gene expression data.
        •        GSEA Results: Identification of significantly enriched pathways and gene sets for upregulated and downregulated genes.


Conclusions
This analysis demonstrates the power of integrating statistical and machine learning methods in bioinformatics to analyze large-scale gene expression data. By leveraging the comprehensive gene expression data available in the TCGA BRCA dataset, significant insights into breast cancer gene expression can be uncovered, potentially aiding in better understanding and treatment of this disease.


Technologies Used
        •        Programming Language: Python
        •        Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, gseapy
        •        Tools: Google Colab for data analysis and visualization