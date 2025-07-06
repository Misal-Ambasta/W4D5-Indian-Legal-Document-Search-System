import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import json

# Streamlit configuration
st.set_page_config(
    page_title="Indian Legal Document Search System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f4e79;
        font-size: 2.5em;
        margin-bottom: 30px;
        font-weight: bold;
    }
    .method-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1f4e79;
    }
    .metric-box {
        background: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 5px;
    }
    .result-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #ddd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"

def call_api(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None) -> Dict:
    """Make API call to FastAPI backend"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files)
            else:
                response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return {}
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Please make sure FastAPI server is running on http://localhost:8000")
        return {}
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return {}

def display_search_results(results: List[Dict], method_name: str):
    """Display search results for a specific method"""
    st.subheader(f"🔍 {method_name}")
    
    if not results:
        st.write("No results found")
        return
    
    for i, result in enumerate(results, 1):
        doc = result['document']
        similarity = result['similarity']
        
        with st.container():
            st.markdown(f"""
            <div class="result-card">
                <h4>#{i} {doc['title']}</h4>
                <p><strong>Category:</strong> {doc['category']} | <strong>Act:</strong> {doc['act']} | <strong>Section:</strong> {doc['section']}</p>
                <p><strong>Similarity Score:</strong> {similarity:.4f}</p>
                <p><strong>Content:</strong> {doc['content'][:200]}...</p>
            </div>
            """, unsafe_allow_html=True)

def display_metrics(metrics: Dict[str, float], method_name: str):
    """Display metrics for a method"""
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown(f"""
        <div class="metric-box">
            <h4>Precision</h4>
            <h2>{metrics.get('precision', 0):.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div class="metric-box">
            <h4>Recall</h4>
            <h2>{metrics.get('recall', 0):.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
        <div class="metric-box">
            <h4>F1-Score</h4>
            <h2>{metrics.get('f1_score', 0):.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        if 'diversity_score' in metrics:
            st.markdown(f"""
            <div class="metric-box">
                <h4>Diversity</h4>
                <h2>{metrics['diversity_score']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-box">
                <h4>-</h4>
                <h2>N/A</h2>
            </div>
            """, unsafe_allow_html=True)

def create_comparison_chart(methods: List[Dict]):
    """Create comparison chart for all methods"""
    method_names = [method['method'] for method in methods]
    precision_scores = [method['metrics']['precision'] for method in methods]
    recall_scores = [method['metrics']['recall'] for method in methods]
    f1_scores = [method['metrics']['f1_score'] for method in methods]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Precision',
        x=method_names,
        y=precision_scores,
        marker_color='#1f4e79'
    ))
    
    fig.add_trace(go.Bar(
        name='Recall',
        x=method_names,
        y=recall_scores,
        marker_color='#4a90e2'
    ))
    
    fig.add_trace(go.Bar(
        name='F1-Score',
        x=method_names,
        y=f1_scores,
        marker_color='#7bb3f0'
    ))
    
    fig.update_layout(
        title='Method Comparison - Performance Metrics',
        xaxis_title='Similarity Methods',
        yaxis_title='Score',
        barmode='group',
        height=400
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">⚖️ Indian Legal Document Search System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Check API health
    health = call_api("/health")
    if health:
        st.sidebar.success(f"✅ API Connected")
        st.sidebar.info(f"📄 {health.get('total_documents', 0)} documents loaded")
    else:
        st.sidebar.error("❌ API Not Connected")
        st.sidebar.warning("Please start the FastAPI server first")
        return
    
    # Navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Search & Compare", "Upload Documents", "Performance Analysis", "About"]
    )
    
    if page == "Search & Compare":
        st.header("🔍 Search & Compare Methods")
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Enter your search query:",
                placeholder="e.g., Income tax deduction for education",
                help="Enter a legal query to search across all documents"
            )
        
        with col2:
            top_k = st.selectbox("Results per method:", [3, 5, 10], index=1)
        
        # Predefined test queries
        st.subheader("📋 Test Queries")
        test_queries = [
            "Income tax deduction for education",
            "GST rate for textile products",
            "Property registration process",
            "Court fee structure"
        ]
        
        query_cols = st.columns(4)
        for i, test_query in enumerate(test_queries):
            with query_cols[i]:
                if st.button(f"📝 {test_query}", key=f"test_{i}"):
                    query = test_query
                    st.rerun()
        
        # Search execution
        if query:
            with st.spinner("Searching with all methods..."):
                search_data = {
                    "query": query,
                    "top_k": top_k
                }
                
                results = call_api("/search", method="POST", data=search_data)
                
                if results:
                    st.success(f"✅ Search completed for: '{query}'")
                    
                    # Display summary
                    summary = results.get('summary', {})
                    st.subheader("📊 Search Summary")
                    
                    summary_cols = st.columns(4)
                    with summary_cols[0]:
                        st.metric("Total Documents", summary.get('total_documents', 0))
                    with summary_cols[1]:
                        st.metric("Avg Precision", f"{summary.get('avg_precision', 0):.3f}")
                    with summary_cols[2]:
                        st.metric("Avg Recall", f"{summary.get('avg_recall', 0):.3f}")
                    with summary_cols[3]:
                        st.metric("Avg F1-Score", f"{summary.get('avg_f1', 0):.3f}")
                    
                    # Method comparison tabs
                    methods = results.get('methods', [])
                    if methods:
                        # Create comparison chart
                        st.subheader("📈 Performance Comparison")
                        comparison_chart = create_comparison_chart(methods)
                        st.plotly_chart(comparison_chart, use_container_width=True)
                        
                        # Display results in tabs
                        st.subheader("🔍 Detailed Results Comparison")
                        
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "Cosine Similarity", 
                            "Euclidean Distance", 
                            "MMR", 
                            "Hybrid Similarity"
                        ])
                        
                        with tab1:
                            method_data = methods[0]
                            display_metrics(method_data['metrics'], method_data['method'])
                            display_search_results(method_data['results'], method_data['method'])
                        
                        with tab2:
                            method_data = methods[1]
                            display_metrics(method_data['metrics'], method_data['method'])
                            display_search_results(method_data['results'], method_data['method'])
                        
                        with tab3:
                            method_data = methods[2]
                            display_metrics(method_data['metrics'], method_data['method'])
                            display_search_results(method_data['results'], method_data['method'])
                        
                        with tab4:
                            method_data = methods[3]
                            display_metrics(method_data['metrics'], method_data['method'])
                            display_search_results(method_data['results'], method_data['method'])
                        
                        # Side-by-side comparison
                        st.subheader("🔄 Side-by-Side Comparison")
                        
                        cols = st.columns(4)
                        
                        for i, method_data in enumerate(methods):
                            with cols[i]:
                                st.markdown(f"**{method_data['method']}**")
                                
                                # Show top 3 results
                                for j, result in enumerate(method_data['results'][:3]):
                                    doc = result['document']
                                    similarity = result['similarity']
                                    
                                    st.markdown(f"""
                                    <div style="border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 5px;">
                                        <h6>#{j+1} {doc['title'][:30]}...</h6>
                                        <p><small><strong>Score:</strong> {similarity:.3f}</small></p>
                                        <p><small><strong>Category:</strong> {doc['category']}</small></p>
                                    </div>
                                    """, unsafe_allow_html=True)
    
    elif page == "Upload Documents":
        st.header("📤 Upload Legal Documents")
        
        st.info("Upload PDF, DOCX, or TXT files containing legal documents to expand the search database.")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size} bytes",
                "File type": uploaded_file.type
            }
            
            st.subheader("📄 File Information")
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
            
            # Upload button
            if st.button("Upload Document", type="primary"):
                with st.spinner("Uploading and processing document..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    result = call_api("/upload", method="POST", files=files)
                    
                    if result:
                        st.success(f"✅ {result.get('message', 'Document uploaded successfully')}")
                        st.balloons()
                        
                        # Refresh document count
                        health = call_api("/health")
                        if health:
                            st.info(f"📄 Total documents: {health.get('total_documents', 0)}")
        
        # Display current documents
        st.subheader("📚 Current Document Library")
        
        docs_result = call_api("/documents")
        if docs_result and 'documents' in docs_result:
            documents = docs_result['documents']
            
            # Create document summary
            df_docs = pd.DataFrame([
                {
                    "ID": doc['id'],
                    "Title": doc['title'],
                    "Category": doc['category'],
                    "Act": doc['act'],
                    "Section": doc['section'],
                    "Content Length": len(doc['content'])
                }
                for doc in documents
            ])
            
            st.dataframe(df_docs, use_container_width=True)
            
            # Document statistics
            st.subheader("📊 Document Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Category distribution
                category_counts = df_docs['Category'].value_counts()
                fig_cat = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Documents by Category"
                )
                st.plotly_chart(fig_cat, use_container_width=True)
            
            with col2:
                # Content length distribution
                fig_len = px.histogram(
                    df_docs,
                    x='Content Length',
                    title="Document Length Distribution",
                    nbins=10
                )
                st.plotly_chart(fig_len, use_container_width=True)
    
    elif page == "Performance Analysis":
        st.header("📊 Performance Analysis")
        
        st.info("Analyze and compare the performance of different similarity methods across various test queries.")
        
        # Run batch analysis
        if st.button("🚀 Run Comprehensive Analysis", type="primary"):
            test_queries = [
                "Income tax deduction for education",
                "GST rate for textile products",
                "Property registration process",
                "Court fee structure"
            ]
            
            analysis_results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, query in enumerate(test_queries):
                status_text.text(f"Analyzing query: {query}")
                
                search_data = {"query": query, "top_k": 5}
                result = call_api("/search", method="POST", data=search_data)
                
                if result:
                    analysis_results.append({
                        "query": query,
                        "result": result
                    })
                
                progress_bar.progress((i + 1) / len(test_queries))
            
            status_text.text("Analysis complete!")
            
            if analysis_results:
                st.success("✅ Analysis completed successfully!")
                
                # Create comprehensive comparison
                st.subheader("📈 Method Performance Across All Queries")
                
                # Prepare data for visualization
                performance_data = []
                
                for analysis in analysis_results:
                    query = analysis['query']
                    methods = analysis['result']['methods']
                    
                    for method in methods:
                        performance_data.append({
                            "Query": query,
                            "Method": method['method'],
                            "Precision": method['metrics']['precision'],
                            "Recall": method['metrics']['recall'],
                            "F1-Score": method['metrics']['f1_score']
                        })
                
                df_performance = pd.DataFrame(performance_data)
                
                # Average performance by method
                avg_performance = df_performance.groupby('Method')[['Precision', 'Recall', 'F1-Score']].mean()
                
                st.subheader("🏆 Average Performance Rankings")
                st.dataframe(avg_performance.round(4), use_container_width=True)
                
                # Performance across queries
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_precision = px.bar(
                        df_performance,
                        x='Method',
                        y='Precision',
                        color='Query',
                        title="Precision by Method and Query",
                        barmode='group'
                    )
                    st.plotly_chart(fig_precision, use_container_width=True)
                
                with col2:
                    fig_f1 = px.bar(
                        df_performance,
                        x='Method',
                        y='F1-Score',
                        color='Query',
                        title="F1-Score by Method and Query",
                        barmode='group'
                    )
                    st.plotly_chart(fig_f1, use_container_width=True)
                
                # Method comparison radar chart
                st.subheader("🎯 Method Comparison Radar Chart")
                
                fig_radar = go.Figure()
                
                methods_list = df_performance['Method'].unique()
                metrics = ['Precision', 'Recall', 'F1-Score']
                
                for method in methods_list:
                    method_data = df_performance[df_performance['Method'] == method][metrics].mean()
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=method_data.values,
                        theta=metrics,
                        fill='toself',
                        name=method
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Method Performance Comparison"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Recommendations")
                
                best_precision = avg_performance['Precision'].idxmax()
                best_recall = avg_performance['Recall'].idxmax()
                best_f1 = avg_performance['F1-Score'].idxmax()
                
                st.markdown(f"""
                <div class="method-card">
                    <h4>📊 Performance Summary</h4>
                    <ul>
                        <li><strong>Best Precision:</strong> {best_precision} ({avg_performance.loc[best_precision, 'Precision']:.3f})</li>
                        <li><strong>Best Recall:</strong> {best_recall} ({avg_performance.loc[best_recall, 'Recall']:.3f})</li>
                        <li><strong>Best F1-Score:</strong> {best_f1} ({avg_performance.loc[best_f1, 'F1-Score']:.3f})</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Method-specific insights
                st.markdown("""
                <div class="method-card">
                    <h4>🔍 Method Insights</h4>
                    <ul>
                        <li><strong>Cosine Similarity:</strong> Good for semantic similarity, works well with transformer embeddings</li>
                        <li><strong>Euclidean Distance:</strong> Geometric distance measure, may be sensitive to high-dimensional spaces</li>
                        <li><strong>MMR:</strong> Reduces redundancy, provides diverse results but may sacrifice some relevance</li>
                        <li><strong>Hybrid Similarity:</strong> Combines semantic and legal entity matching for domain-specific improvements</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    elif page == "About":
        st.header("ℹ️ About the System")
        
        st.markdown("""
        ## 🎯 Indian Legal Document Search System
        
        This system implements and compares four different similarity methods for searching Indian legal documents:
        
        ### 🔍 Similarity Methods
        
        1. **Cosine Similarity**
           - Measures the cosine of the angle between document vectors
           - Best for semantic similarity in high-dimensional spaces
           - Uses sentence transformer embeddings
        
        2. **Euclidean Distance**
           - Calculates geometric distance between document vectors
           - Converted to similarity score using inverse transformation
           - Good for spatial relationships in embedding space
        
        3. **Maximum Marginal Relevance (MMR)**
           - Balances relevance and diversity in results
           - Reduces redundancy by considering similarity to already selected documents
           - Formula: λ × relevance - (1-λ) × diversity
        
        4. **Hybrid Similarity**
           - Combines semantic similarity with legal entity matching
           - Formula: 0.6 × cosine_similarity + 0.4 × entity_similarity
           - Optimized for legal document domain
        
        ### 📊 Evaluation Metrics
        
        - **Precision**: Relevant documents in top results
        - **Recall**: Coverage of all relevant documents
        - **F1-Score**: Harmonic mean of precision and recall
        - **Diversity Score**: Variety in result categories (for MMR)
        
        ### 🏗️ Technical Architecture
        
        - **Backend**: FastAPI with sentence-transformers
        - **Frontend**: Streamlit for interactive UI
        - **ML Models**: all-MiniLM-L6-v2 for embeddings
        - **Document Processing**: PDF, DOCX, TXT support
        
        ### 📚 Test Dataset
        
        The system includes sample documents from:
        - Income Tax Act sections
        - GST Act provisions
        - Property law documents
        - Court fee structures
        
        ### 🚀 Usage
        
        1. **Search**: Enter queries and compare results across all methods
        2. **Upload**: Add new legal documents to expand the database
        3. **Analysis**: Run comprehensive performance analysis
        4. **Compare**: View side-by-side comparisons of different methods
        
        ### 🔧 Setup Instructions
        
        1. Install dependencies: `pip install -r requirements.txt`
        2. Start FastAPI server: `uvicorn main:app --reload`
        3. Run Streamlit app: `streamlit run app.py`
        4. Access at: http://localhost:8501
        
        ### 📈 Performance Insights
        
        Based on testing with legal queries:
        - **Hybrid method** often performs best for domain-specific queries
        - **Cosine similarity** provides consistent semantic matching
        - **MMR** is valuable when diversity is important
        - **Euclidean distance** may be less effective in high-dimensional spaces
        
        ---
        
        **Note**: This system is designed for educational and research purposes. 
        Always consult legal professionals for actual legal advice.
        """)

if __name__ == "__main__":
    main()