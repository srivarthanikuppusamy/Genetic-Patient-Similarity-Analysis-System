import streamlit as st
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from model_utils import PatientSimilarityPipeline

# Page Configuration
st.set_page_config(page_title="Genetic Patient Similarity System", layout="wide")
st.title("🧬 Genetic Patient Similarity Analysis System")
st.markdown("Investigate natural patient groups based on genetic and clinical features using **Ward's Hierarchical Agglomerative Clustering**.")

# Initialize the ML Pipeline state
if 'pipeline' not in st.session_state:
    st.session_state['pipeline'] = PatientSimilarityPipeline()

pipeline = st.session_state['pipeline']

# Sidebar - Dataset Upload
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Patient Genomes CSV", type=['csv'])

if uploaded_file is not None:
    # Check if a new file has been uploaded
    if 'data_uploaded' not in st.session_state or st.session_state.get('uploaded_filename') != uploaded_file.name:
        with st.spinner("Preprocessing data and computing hierarchy linkage..."):
            try:
                df = pd.read_csv(uploaded_file)
                
                # Execute in-memory processing pipeline
                X_scaled, df_clean = pipeline.preprocess_data(df)
                linkage_mat = pipeline.compute_linkage()
                
                # Cache results in Session State
                st.session_state['data_uploaded'] = True
                st.session_state['uploaded_filename'] = uploaded_file.name
                st.session_state['raw_df'] = df_clean
                st.session_state['X_scaled'] = X_scaled
                st.session_state['linkage_mat'] = linkage_mat
                st.sidebar.success("Data processed successfully!")
            except Exception as e:
                st.sidebar.error(f"Error processing file: {e}")

# Render Dashboard Controls and Visualizations
if st.session_state.get('data_uploaded'):
    linkage_mat = st.session_state['linkage_mat']

    st.subheader("Patient Similarity Dashboard")
    
    # 1. Dendrogram Tree Plot
    st.markdown("### Hierarchical Clustering Dendrogram")
    st.markdown("Visualizing how patient clusters merge bottom-up based on genetic distance.")
    
    fig_dendro = ff.create_dendrogram(linkage_mat, orientation='left')
    fig_dendro.update_layout(
        width=1000, 
        height=500, 
        xaxis_title="Distance (Ward's Linkage)", 
        yaxis_title="Patient Indices"
    )
    st.plotly_chart(fig_dendro, use_container_width=True)

    st.divider()

    # 2. Interactive Grouping Setup
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Grouping Controls")
        n_clusters = st.selectbox("Number of Groups:", options=list(range(2, 11)), index=1)
        btn_generate = st.button("Generate Groups")

    if btn_generate or 'hier_labels' in st.session_state:
        if btn_generate or st.session_state.get('current_n') != n_clusters:
            hier_labels, hier_metrics = pipeline.fit_hierarchical_clustering(n_clusters)
            st.session_state['hier_labels'] = hier_labels
            st.session_state['hier_metrics'] = hier_metrics
            st.session_state['current_n'] = n_clusters

        hier_metrics = st.session_state['hier_metrics']
        hier_labels = st.session_state['hier_labels']

        with col2:
            st.markdown("### Group Information & Quality Metrics")
            m = hier_metrics
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Silhouette Score", f"{m['silhouette']:.3f}")
            mc2.metric("Davies-Bouldin Index", f"{m['davies_bouldin']:.3f}")
            mc3.metric("Calinski-Harabasz Index", f"{m['calinski_harabasz']:.1f}")

            # Distribution Chart
            df_result = st.session_state['raw_df'].copy()
            df_result['Cluster'] = hier_labels
            cluster_counts = df_result['Cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Patient Count']
            
            fig_dist = px.bar(
                cluster_counts, 
                x='Cluster', 
                y='Patient Count', 
                title="Patient Distribution across Clusters", 
                color='Cluster'
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # 3. Critical Thinking Challenge Section
    st.subheader("💡 Critical Thinking Challenge: Ward's Hierarchical vs. K-Means")
    
    if st.button("Run Comparison Analysis"):
        current_n = st.session_state.get('current_n', n_clusters)
        hier_labels, hier_metrics = pipeline.fit_hierarchical_clustering(current_n)
        kmeans_labels, kmeans_metrics = pipeline.fit_kmeans_clustering(current_n)
        
        comp_df = pd.DataFrame({
            'Metric': [
                'Silhouette Score (Higher = Better)', 
                'Davies-Bouldin (Lower = Better)', 
                'Calinski-Harabasz (Higher = Better)'
            ],
            "Ward's Hierarchical": [
                hier_metrics['silhouette'], 
                hier_metrics['davies_bouldin'], 
                hier_metrics['calinski_harabasz']
            ],
            'K-Means': [
                kmeans_metrics['silhouette'], 
                kmeans_metrics['davies_bouldin'], 
                kmeans_metrics['calinski_harabasz']
            ]
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
