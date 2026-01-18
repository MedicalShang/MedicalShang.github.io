---
title: "TCGA-BRCA Multi-Omics Analysis"
collection: portfolio
permalink: /portfolio/tcga-brca-analysis/
date: 2025-01-01
excerpt: "Integrative analysis of TCGA-BRCA clinical and transcriptomic data using PCA-based dimensionality reduction and logistic regression modeling."
image: /images/portfolio/tcga-brca-analysis/thumbnail.png
---

## Project Background

Breast cancer is a highly heterogeneous disease. This project explores transcriptomic and clinical data from the TCGA-BRCA cohort to investigate subtype discrimination using classical machine learning methods.

## Data & Methods

- **Dataset**: TCGA-BRCA (RNA-seq + clinical)
- **Samples**: 1,084 patients, 959 retained after QC
- **Features**: 20,531 genes
- **Dimensionality reduction**: PCA (95% variance, 654 components)
- **Model**: Logistic regression (stratified 80/20 split)

## Key Results

<p align="center">
  <img src="/images/portfolio/tcga-brca-analysis/age_distribution.png" width="480">
</p>

<p align="center">
  <img src="/images/portfolio/tcga-brca-analysis/pca_projection.png" width="520">
</p>

## Conclusion & Future Work

This project demonstrates the feasibility and limitations of classical machine learning approaches for breast cancer subtype classification, highlighting challenges related to class imbalance and translational relevance.
