import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px

API_URL = "http://localhost:5000/api"

st.set_page_config(page_title="Genetic Patient Similarity System", layout="wide")
st.title("🧬 Genetic Patient Similarity Analysis System")
st.markdown("Investigate natural patient groups based on genetic and clinical features using **Ward's Hierarchical Agglomerative Clustering**.")

# Sidebar - Dataset Input
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Patient Genomes CSV", type=['csv'])

if uploaded_file is not None:
    if 'data_uploaded' not in st.session_state:
        with st.spinner("Preprocessing data and computing hierarchy linkage..."):
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
            res = requests.post(f"{API_URL}/upload", files=files)
            if res.status_code == 200:
                st.session_state['data_uploaded'] = True
                st.session_state['upload_res'] = res.json()
                st.sidebar.success("Data processed successfully!")
            else:
                st.error("Failed to process dataset.")

if st.session_state.get('data_uploaded'):
    upload_data = st.session_state['upload_res']
    linkage_mat = np.array(upload_data['linkage_matrix'])

    st.subheader("Patient Similarity Dashboard")
    
    # 1. Dendrogram Visualization
    st.markdown("### Hierarchical Clustering Dendrogram")
    st.markdown("Visualizing how patient clusters merge bottom-up based on genetic distance.")
    
    fig_dendro = ff.create_dendrogram(linkage_mat, orientation='left')
    fig_dendro.update_layout(width=1000, height=500, xaxis_title="Distance (Ward's Linkage)", yaxis_title="Patient Indices")
    st.plotly_chart(fig_dendro, use_container_width=True)

    st.divider()

    # 2. Interactive Grouping Setup
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Grouping Controls")
        n_clusters = st.selectbox("Number of Groups:", options=list(range(2, 11)), index=1)
        btn_generate = st.button("Generate Groups")

    if btn_generate or 'cluster_res' in st.session_state:
        if btn_generate:
            res = requests.post(f"{API_URL}/cluster", json={"n_clusters": n_clusters})
            st.session_state['cluster_res'] = res.json()
        
        cluster_data = st.session_state['cluster_res']

        with col2:
            st.markdown("### Group Information & Quality Metrics")
            m = cluster_data['metrics']
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Silhouette Score", f"{m['silhouette']:.3f}")
            mc2.metric("Davies-Bouldin Index", f"{m['davies_bouldin']:.3f}")
            mc3.metric("Calinski-Harabasz Index", f"{m['calinski_harabasz']:.1f}")

            # Distribution Chart
            dist_df = pd.DataFrame(list(cluster_data['cluster_distribution'].items()), columns=['Cluster', 'Patient Count'])
            fig_dist = px.bar(dist_df, x='Cluster', y='Patient Count', title="Patient Distribution across Clusters", color='Cluster')
            st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # 3. Critical Thinking Challenge Section
    st.subheader("💡 Critical Thinking Challenge: Ward's Hierarchical vs. K-Means")
    
    if st.button("Run Comparison Analysis"):
        comp_res = requests.post(f"{API_URL}/compare", json={"n_clusters": n_clusters}).json()
        
        h_m = comp_res['hierarchical']['metrics']
        k_m = comp_res['kmeans']['metrics']
        
        comp_df = pd.DataFrame({
            'Metric': ['Silhouette Score (Higher = Better)', 'Davies-Bouldin (Lower = Better)', 'Calinski-Harabasz (Higher = Better)'],
            "Ward's Hierarchical": [h_m['silhouette'], h_m['davies_bouldin'], h_m['calinski_harabasz']],
            'K-Means': [k_m['silhouette'], k_m['davies_bouldin'], k_m['calinski_harabasz']]
        })
        
        st.table(comp_df)

        st.markdown("""
        #### Why Algorithm Structures Differ on Genetic & Clinical Features:
        1. **Objective Function & Geometry:**
           - **K-Means** optimizes for variance around spherical centroids (Euclidean center of mass). It forces clusters into convex, globular forms of equal size.
           - **Ward's Hierarchical Clustering** minimizes the increase in total within-cluster variance when merging groups recursively. It respects non-globular and hierarchical relationships present in complex genetic mutations.
        2. **Determinism vs. Random Initialization:**
           - K-Means relies on initial centroid seeds (`k-means++`), which can lead to localized convergence depending on seed choices.
           - Ward's algorithm produces **deterministic results** directly derived from the distance hierarchy without requiring random initialization.
        3. **Hierarchy vs. Partitioning:**
           - Hierarchical clustering creates nested structures, showing how subgroups (e.g., specific genetic sub-mutations) nest within broader clinical patient groups. K-Means provides a flat partition.
        """)