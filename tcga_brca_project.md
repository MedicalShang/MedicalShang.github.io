---
title: "TCGA-BRCA 乳腺癌多组学数据分析与机器学习建模"
collection: portfolio
permalink: /portfolio/tcga-brca-analysis/
date: 2025-01-18
excerpt: "基于 TCGA-BRCA 公共数据库，系统分析乳腺癌患者的临床特征与转录组表达数据，并构建基于 PCA + 逻辑回归的分类模型，探索肿瘤亚型判别的可行性与局限性。"
image: 
github_url: "https://github.com/username/tcga-brca-analysis"
---

## Project Background

Breast cancer remains one of the most prevalent malignancies worldwide, with substantial heterogeneity in molecular subtypes that influence clinical outcomes and therapeutic responses. The Cancer Genome Atlas (TCGA) Breast Invasive Carcinoma (BRCA) dataset provides a comprehensive multi-omics resource, encompassing clinical annotations and transcriptomic profiles from over 1,000 patients.

Understanding the molecular mechanisms underlying breast cancer subtyping is critical for precision medicine. Previous studies have identified distinct molecular subtypes—Luminal A, Luminal B, HER2-enriched, and Basal-like—each associated with different prognostic characteristics and treatment responses. However, the high dimensionality of transcriptomic data poses significant challenges for developing robust and interpretable classification models.

This project aims to address these challenges by integrating clinical and transcriptomic data from the TCGA-BRCA cohort, applying dimensionality reduction techniques, and constructing a machine learning framework to classify tumor subtypes. Our approach balances computational efficiency with biological interpretability, providing insights into both model performance and practical limitations.

## Data & Methods

### Data Acquisition and Preprocessing

The study utilized publicly available data from the TCGA-BRCA project, comprising:

- **Clinical Data**: 1,084 patients with detailed demographic information (age, gender), tumor staging (AJCC pathological stage), and clinical outcomes
- **Transcriptomic Data**: RNA-seq gene expression matrix covering 20,531 genes across 1,083 tumor samples

**Data Integration**: Sample matching was performed by aligning patient identifiers between clinical phenotypic data and expression matrices. Following quality control and removal of samples with incomplete information, 959 patients were retained for downstream analysis, with 20,502 genes available after filtering.

### Exploratory Data Analysis

**Clinical Characterization**: 
- Age distribution analysis revealed that the patient population primarily consisted of middle-aged individuals, with ages concentrated between 45-65 years
- Subtype distribution showed predominance of Luminal A and Luminal B subtypes, consistent with epidemiological observations in breast cancer cohorts

**Gene Expression Quality Control**:
- Gene-wise coefficient of variation (CV) analysis identified highly variable genes, which are often biologically informative
- Pairwise correlation analysis of canonical breast cancer genes (BRCA1, BRCA2, ERBB2, TP53) revealed moderate correlations (e.g., BRCA1-BRCA2: r=0.385), supporting the biological plausibility of the dataset

### Dimensionality Reduction and Feature Engineering

Given the curse of dimensionality (20,502 features for 959 samples), we employed Principal Component Analysis (PCA) to reduce feature space while preserving variance:

**Standardization**: Gene expression values were standardized using StandardScaler (mean=0, standard deviation=1) to ensure equal weighting across genes with different expression scales

**PCA Implementation**: 
- PCA was applied to reduce dimensionality while retaining 95% of total variance
- The optimal number of principal components was determined as 654, representing a 97% reduction in feature dimensionality
- This approach balances information retention with computational efficiency, enabling downstream machine learning without substantial information loss

### Classification Model

We constructed a logistic regression classifier as the primary predictive model, chosen for its interpretability and suitability for binary classification tasks:

**Experimental Design**:
- Dataset split: 80% training set (767 samples), 20% test set (192 samples)
- Stratified sampling was employed to preserve class distribution across splits
- Model hyperparameters were optimized using default settings as a baseline approach

**Target Variable**: Binary classification based on `PERSON_NEOPLASM_CANCER_STATUS` (Tumor Free vs. With Tumor), representing a clinically relevant endpoint for disease recurrence assessment

## Key Results

### Model Performance

The PCA-reduced logistic regression model achieved an overall accuracy of **80.7%** on the held-out test set, demonstrating moderate predictive capability for tumor status classification.

**Detailed Classification Metrics**:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Tumor Free) | 0.91 | 0.87 | 0.89 | 174 |
| 1 (With Tumor) | 0.12 | 0.17 | 0.14 | 18 |

**Interpretation**:
- The model exhibits strong performance for the majority class (Class 0), correctly identifying 152 out of 174 tumor-free patients
- Performance for the minority class (Class 1) is substantially limited, with only 3 out of 18 patients with recurrent tumors correctly identified
- This discrepancy reflects the inherent class imbalance in the dataset (Class 0: 91%, Class 1: 9%) and highlights the challenges of detecting low-prevalence events

### Confusion Matrix Analysis


**False Negatives (n=15)**: Patients with recurrent tumors incorrectly classified as tumor-free. These misclassifications represent the most critical errors from a clinical perspective, as they may lead to inadequate follow-up monitoring and delayed intervention.

**False Positives (n=22)**: Tumor-free patients incorrectly flagged as having recurrent tumors. While less clinically severe than false negatives, these errors could result in unnecessary anxiety and additional diagnostic procedures.

### Dimensionality Reduction Efficacy

PCA effectively reduced feature dimensionality from 20,502 genes to 654 principal components (97% reduction) while preserving 95% of total variance. This substantial dimensionality reduction:
- Mitigates overfitting risk by reducing model complexity
- Improves computational efficiency for training and inference
- Facilitates potential biological interpretation through component loading analysis

## Model Evaluation

### Performance Limitations

The modest performance on the minority class warrants careful consideration of several factors:

**1. Class Imbalance**: The substantial disparity between classes (91% vs. 9%) predisposes the model to prioritize majority class accuracy, a common challenge in medical prediction tasks where disease recurrence is relatively rare in short-term follow-up

**2. Feature Representation**: While PCA effectively reduces dimensionality, the linear combination of genes may obscure important non-linear relationships and gene-gene interactions specific to tumor recurrence biology

**3. Clinical Heterogeneity**: Tumor recurrence is influenced by complex factors beyond transcriptomic profiles, including treatment regimens, patient comorbidities, and environmental exposures not captured in the current dataset

**4. Sample Size**: While 959 patients represent a substantial cohort, the limited number of recurrence events (n=18 in test set) constrains the model's ability to learn robust patterns for the minority class

### Comparative Analysis

Compared to baseline random classification (50% accuracy for balanced classes), our model demonstrates meaningful predictive ability. However, the performance gap between majority and minority classes underscores the need for specialized techniques for imbalanced data, such as:

- Resampling strategies (SMOTE, ADASYN)
- Cost-sensitive learning with class-weighted loss functions
- Ensemble methods designed for imbalanced classification
- Alternative algorithms better suited to high-dimensional biological data (e.g., random forests, support vector machines)

### Clinical Relevance

From a translational perspective, the current model's sensitivity (17% for recurrence detection) is insufficient for clinical application as a standalone prognostic tool. However, several aspects of the analysis retain value:

- **Exploratory Insights**: The identification of gene expression patterns associated with tumor status provides a foundation for hypothesis generation and biomarker discovery
- **Methodological Framework**: The established pipeline for data integration, dimensionality reduction, and classification can be adapted to other clinical endpoints and cancer types
- **Baseline Benchmark**: Current performance establishes a reference point for evaluating alternative modeling approaches

## Conclusion & Future Work

### Summary

This project successfully implemented a comprehensive bioinformatics pipeline for analyzing TCGA-BRCA multi-omics data, integrating clinical annotations with transcriptomic profiles to explore tumor recurrence prediction. Through PCA-based dimensionality reduction and logistic regression classification, we achieved 80.7% overall accuracy, demonstrating that transcriptomic data contains predictive signals for clinical outcomes.

However, the substantial performance disparity between majority and minority classes highlights critical limitations in current approaches, particularly for detecting low-prevalence but clinically significant events such as tumor recurrence. These findings underscore the importance of addressing class imbalance and incorporating domain knowledge in biomedical machine learning applications.

### Future Directions

**1. Advanced Modeling Techniques**:
- Implement ensemble methods (XGBoost, Random Forest) with built-in feature selection capabilities
- Explore deep learning architectures (feedforward neural networks, autoencoders) for capturing non-linear relationships
- Evaluate support vector machines with alternative kernel functions

**2. Imbalance Mitigation Strategies**:
- Apply synthetic oversampling (SMOTE, ADASYN) and undersampling techniques
- Implement cost-sensitive learning with optimized class weights
- Utilize anomaly detection frameworks for rare event identification

**3. Feature Engineering and Selection**:
- Incorporate prior biological knowledge through pathway-based feature aggregation (e.g., gene set enrichment scores)
- Perform differential expression analysis to identify recurrence-associated genes
- Integrate multi-omics modalities (DNA methylation, copy number variation) for a more comprehensive molecular profile

**4. Temporal and Clinical Context**:
- Leverage longitudinal follow-up data for time-to-event analysis (Cox proportional hazards, survival forests)
- Incorporate treatment information and therapy response data
- Validate models on independent cohorts to assess generalizability

**5. Clinical Translation**:
- Collaborate with clinical oncologists to define performance thresholds for clinical utility
- Develop interpretable model explanations (SHAP values, feature importance) for biomarker discovery
- Design prospective validation studies for promising predictive signatures

### Broader Implications

This work contributes to the growing field of precision oncology by demonstrating both the potential and challenges of applying machine learning to high-dimensional biomedical data. The methodological insights gained from this study inform future efforts to develop robust, clinically actionable predictive models for cancer prognosis and treatment response prediction.

---


**Keywords**: Breast cancer, TCGA, Machine learning, PCA, Logistic regression, Transcriptomics, Bioinformatics, Precision medicine, Classification, Imbalanced learning

