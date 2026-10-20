# 🧬 Genetic Patient Similarity Analysis System

An end-to-end Machine Learning web application designed to investigate how patients naturally group based on genetic and clinical attributes. The system utilizes **Ward's Hierarchical Agglomerative Clustering** to establish patient relationships and presents an interactive **Streamlit Dashboard** for visualization and comparative analysis.

🚀 **Live Interactive App:** [Genetic Patient Similarity Analysis System](https://genetic-patient-similarity-analysis-system-5fyygdfzchqjcoiu9zl.streamlit.app/)

---

## 📌 Project Overview

Unlike traditional supervised classification tasks where patient outcomes or diagnoses are predetermined, healthcare researchers often need to explore how patient cohorts form subgroups organically. 

This project solves this challenge by:
1. Standardizing and encoding complex genetic/clinical patient features.
2. Building an explicit distance-based hierarchical tree.
3. Allowing researchers to slice patient cohorts dynamically at different similarity levels.
4. Comparing hierarchical structure against traditional partitioning algorithms (K-Means).

---

## 🏗️ System Architecture

The application runs in-memory using Streamlit to perform real-time data cleaning, distance matrix calculation, and clustering validation metrics:

```text
┌─────────────────────────────────────────────────────────────┐
│              Streamlit Web Application                      │
│   (https://genetic-patient-similarity-analysis-system...)   │
│                                                             │
│   - Patient Genome CSV Upload                               │
│   - Plotly Hierarchical Dendrogram Plot                     │
│   - Interactive Grouping Controls (N = 2 to 10)             │
│   - Live Metrics (Silhouette, Davies-Bouldin, Calinski-H.)  │
│   - Side-by-Side Algorithm Comparison (Ward's vs K-Means)   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Imports & Executes
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 ML Pipeline Core                            │
│                 (model_utils.py)                            │
│                                                             │
│   - String Missing Value Cleaning                           │
│   - ColumnTransformer (StandardScaler + OneHotEncoder)      │
│   - Scipy Linkage Matrix (Ward's Minimum Variance)          │
│   - Agglomerative & K-Means Clustering Models               │
└─────────────────────────────────────────────────────────────┘
