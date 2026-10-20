from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from model_utils import PatientSimilarityPipeline

app = Flask(__name__)
pipeline = PatientSimilarityPipeline()
data_store = {}

@app.route('/api/upload', methods=['POST'])
def upload_data():
    """Uploads patient data CSV and performs pre-processing and linkage calculation."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    df = pd.read_csv(file)
    
    X_scaled, df_clean = pipeline.preprocess_data(df)
    linkage_mat = pipeline.compute_linkage()
    
    data_store['raw_df'] = df_clean
    
    return jsonify({
        "status": "success",
        "sample_count": X_scaled.shape[0],
        "feature_count": X_scaled.shape[1],
        "linkage_matrix": linkage_mat.tolist()
    })

@app.route('/api/cluster', methods=['POST'])
def generate_clusters():
    """Generates patient groups using Ward's Hierarchical Agglomerative Clustering."""
    data = request.get_json()
    n_clusters = int(data.get('n_clusters', 3))
    
    hier_labels, hier_metrics = pipeline.fit_hierarchical_clustering(n_clusters)
    
    # Generate per-cluster summaries
    df_result = data_store['raw_df'].copy()
    df_result['Cluster'] = hier_labels
    cluster_summary = df_result.groupby('Cluster').size().to_dict()

    return jsonify({
        "n_clusters": n_clusters,
        "metrics": hier_metrics,
        "cluster_distribution": cluster_summary,
        "labels": hier_labels.tolist()
    })

@app.route('/api/compare', methods=['POST'])
def compare_algorithms():
    """Compares Ward's Hierarchical vs. K-Means performance."""
    data = request.get_json()
    n_clusters = int(data.get('n_clusters', 3))
    
    hier_labels, hier_metrics = pipeline.fit_hierarchical_clustering(n_clusters)
    kmeans_labels, kmeans_metrics = pipeline.fit_kmeans_clustering(n_clusters)
    
    return jsonify({
        "n_clusters": n_clusters,
        "hierarchical": {
            "metrics": hier_metrics,
            "labels": hier_labels.tolist()
        },
        "kmeans": {
            "metrics": kmeans_metrics,
            "labels": kmeans_labels.tolist()
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)