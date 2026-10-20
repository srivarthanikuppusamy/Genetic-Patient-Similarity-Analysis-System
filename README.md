# 🧬 Genetic Patient Similarity Analysis System

An end-to-end Machine Learning web application designed to investigate how patients naturally group based on genetic and clinical attributes. The system utilizes **Ward's Hierarchical Agglomerative Clustering** to establish patient relationships, serves predictions via a **Flask RESTful API**, and offers an interactive **Streamlit Dashboard** for visualization and comparative analysis.

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

The project is structured with a clean separation of concerns between backend computation and frontend visualization:

```text
┌─────────────────────────────────────────┐
│        Streamlit Dashboard              │  <-- Interactive UI
│        (streamlit_app.py)               │      - File Upload & Controls
│  [Dendrogram Plot] [Group Selectors]    │      - Metric Tables & Bar Charts
└────────────────────┬────────────────────┘
                     │ HTTP REST API Requests
                     ▼
┌─────────────────────────────────────────┐
│           Flask API Server              │  <-- ML Backend
│             (app.py)                    │      - Processing Pipelines
│      [StandardScaler & Encoders]        │      - Ward's Clustering & Matrix
│      [Metrics Calculation Engine]       │      - K-Means Comparison
└─────────────────────────────────────────┘
