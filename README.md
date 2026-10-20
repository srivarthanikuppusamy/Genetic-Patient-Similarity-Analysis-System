Genetic Patient Similarity Analysis System
Business Problem
A healthcare research organization wants to investigate whether patients naturally form groups based on similarities in their genetic and clinical characteristics.Unlike a classification problem, the organization does not have predefined groups. Researchers want to visualize how patients gradually form groups as their similarity increases.
Project Objective
Develop a Genetic Patient Similarity Analysis System using Hierarchical Agglomerative Clustering.
Required Implementation
The system should:
Perform data preprocessing.
Select relevant features.
Encode categorical variables.
Standardize numerical features.
Apply Ward's Hierarchical Agglomerative Clustering.
Generate a dendrogram.
Investigate different cluster counts.
Calculate suitable clustering metrics.
Interpret the resulting groups.
Create Flask APIs.
Develop a Streamlit visualization interface.
Frontend Requirement
The Streamlit application should contain:
Patient Similarity Dashboard
[Dendrogram]
Number of Groups:
[ Select ]
[Generate Groups]
Group Information
The system should allow users to explore how patients are grouped at different levels.
Critical Thinking Challenge
Apply both K-Means and Hierarchical Clustering to the same selected features. Compare the resulting groups and explain why the two algorithms may produce different structures.
Primary Algorithm
Ward's Hierarchical Agglomerative Clustering
Dataset link : https://www.kaggle.com/datasets/aibuzz/predict-the-genetic-disorders-datasetof-genomes/data
