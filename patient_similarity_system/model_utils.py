import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class PatientSimilarityPipeline:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.preprocessor = None
        self.feature_names = []
        self.linkage_matrix = None
        self.X_scaled = None
        
    def preprocess_data(self, df):
        """Preprocesses genetic and clinical patient data."""
        # Clean dataframe
        df_clean = df.copy().dropna()
        
        # Identify clinical and genetic feature column types
        num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target/ID columns if present in standard Kaggle dataset formats
        drop_cols = ['Patient Id', 'Patient First Name', 'Family Name', 'Father\'s name', 'Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5']
        num_cols = [c for c in num_cols if c not in drop_cols]
        cat_cols = [c for c in cat_cols if c not in drop_cols]
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
            ]
        )
        
        self.X_scaled = self.preprocessor.fit_transform(df_clean[num_cols + cat_cols])
        
        # Capture transformed feature names
        cat_encoded_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
        self.feature_names = num_cols + list(cat_encoded_names)
        
        return self.X_scaled, df_clean

    def compute_linkage(self):
        """Computes Ward's linkage matrix for dendrogram generation."""
        if self.X_scaled is None:
            raise ValueError("Data not preprocessed. Call preprocess_data first.")
        self.linkage_matrix = linkage(self.X_scaled, method='ward')
        return self.linkage_matrix

    def fit_hierarchical_clustering(self, n_clusters):
        """Applies Agglomerative Clustering with Ward linkage."""
        self.n_clusters = n_clusters
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = model.fit_predict(self.X_scaled)
        
        metrics = self.calculate_metrics(self.X_scaled, cluster_labels)
        return cluster_labels, metrics

    def fit_kmeans_clustering(self, n_clusters):
        """Applies K-Means clustering for algorithm comparison."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.X_scaled)
        
        metrics = self.calculate_metrics(self.X_scaled, cluster_labels)
        return cluster_labels, metrics

    @staticmethod
    def calculate_metrics(X, labels):
        """Computes silhouette, Davies-Bouldin, and Calinski-Harabasz scores."""
        if len(np.unique(labels)) < 2:
            return {"silhouette": 0.0, "davies_bouldin": 0.0, "calinski_harabasz": 0.0}
        
        return {
            "silhouette": float(silhouette_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels))
        }