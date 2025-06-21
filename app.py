import streamlit as st

# Set page config as the FIRST Streamlit command
st.set_page_config(
    page_title="Vector Database Benchmarking",
    page_icon="🚀",
    layout="wide"
)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from typing import Dict, List, Optional, Tuple
import importlib
import sys
from pathlib import Path
import hashlib
import time

# Initialize session state for tracking chart updates
if 'chart_update_key' not in st.session_state:
    st.session_state.chart_update_key = 0

if 'last_operation' not in st.session_state:
    st.session_state.last_operation = None

if 'benchmark_completed' not in st.session_state:
    st.session_state.benchmark_completed = False

# benchmarking_path = Path(__file__).parent  / "weaviate_benchmarking"
# sys.path.append(str(benchmarking_path))
# Import database modules - store import errors to display later
import_errors = []
try:
    import faiss_module as faiss_db
except ImportError as e:
    import_errors.append(f"faiss_module: {e}")
    faiss_db = None

try:
    import milvus_module as milvus_db
except ImportError as e:
    import_errors.append(f"milvus_module: {e}")
    milvus_db = None

try:
    import chroma_module as chroma_db
except ImportError as e:
    import_errors.append(f"chroma_module: {e}")
    chroma_db = None

try:
    import weaviate_module as weaviate_db
except ImportError as e:
    import_errors.append(f"weaviate_module: {e}")
    weaviate_db = None

try:
    import qdrant_module as qdrant_db
except ImportError as e:
    import_errors.append(f"qdrant_module: {e}")
    qdrant_db = None

try:
    import pgvector_module as pgvector_db
except ImportError as e:
    import_errors.append(f"pgvector_module: {e}")
    pgvector_db = None

# Database configuration
DATABASES = {
    'faiss': {'module': faiss_db, 'color': '#DDA0DD'},
    'milvus': {'module': milvus_db, 'color': '#4ECDC4'},
    'chroma': {'module': chroma_db, 'color': '#45B7D1'},
    'weaviate': {'module': weaviate_db, 'color': '#96CEB4'},
    'qdrant': {'module': qdrant_db, 'color': '#FFEAA7'},
    'pgvector': {'module': pgvector_db, 'color': '#FF6B6B'},
}

OPERATIONS = ['create', 'query', 'delete']
EMBEDDING_MODELS = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'snowflake-arctic-embed-l-v2.0']
FAISS_INDEX_TYPES = ['IVFFLAT', 'HNSW']
DEFAULT_SIZES = [500, 1000, 2000, 5000]

def generate_chart_key(operation: str, chart_type: str, additional_info: str = "") -> str:
    """Generate a unique key for charts to avoid ID conflicts."""
    base_string = f"{operation}_{chart_type}_{additional_info}_{st.session_state.chart_update_key}"
    return hashlib.md5(base_string.encode()).hexdigest()[:8]

def update_chart_keys():
    """Update the chart key counter to force re-rendering of charts."""
    st.session_state.chart_update_key += 1

def load_json_results(operation: str) -> Optional[Dict]:
    """Load benchmark results from JSON file with new nested structure."""
    filename = f"{operation}.json"
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            st.error(f"Error loading {filename}: {e}")
            return None
    return None

def save_json_results(operation: str, data: Dict):
    """Save benchmark results to JSON file with new nested structure."""
    filename = f"{operation}.json"
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving {filename}: {e}")

def update_results_structure(operation: str, embedding_model: str, size: int, db_name: str, value: float):
    """Update the results JSON with the new nested structure."""
    # Load existing results or create empty structure
    existing_data = load_json_results(operation) or {}
    
    # Ensure nested structure exists
    if embedding_model not in existing_data:
        existing_data[embedding_model] = {}
    
    if str(size) not in existing_data[embedding_model]:
        existing_data[embedding_model][str(size)] = {}
    
    # Update the value
    existing_data[embedding_model][str(size)][db_name] = value
    
    # Save updated results
    save_json_results(operation, existing_data)

def get_available_datasets(selected_model: str) -> List[str]:
    """Get list of available CSV datasets that match the selected embedding model name."""
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    matching_datasets = []

    for csv_file in csv_files:
        # Check if filename ends with _{selected_model}.csv
        if csv_file.endswith(f"_{selected_model}.csv"):
            matching_datasets.append(csv_file)

    return matching_datasets if matching_datasets else ['No matching datasets found']

def get_available_databases() -> List[str]:
    """Get list of databases that were successfully imported."""
    available = []
    for db_name, config in DATABASES.items():
        if config['module'] is not None:
            available.append(db_name)
    return available

def validate_inputs(selected_dbs: List[str], operation: str, faiss_index: str, selected_sizes: List[int]) -> Tuple[bool, str]:
    """Validate user inputs and return validation status and message."""
    if not selected_dbs:
        return False, "Please select at least one database."
    
    if not selected_sizes:
        return False, "Please select at least one data size."
    
    if 'faiss' in selected_dbs and faiss_index == 'HNSW' and operation == 'delete':
        return False, "FAISS HNSW index does not support delete operations."
    
    # Check if selected databases are available
    available_dbs = get_available_databases()
    unavailable_dbs = [db for db in selected_dbs if db not in available_dbs]
    if unavailable_dbs:
        return False, f"Selected databases not available: {', '.join(unavailable_dbs)}"
    
    return True, ""

def call_database_function(db_name: str, operation: str, **kwargs):
    """Dynamically call the appropriate database function."""
    try:
        module = DATABASES[db_name]['module']
        
        if module is None:
            st.error(f"Module for {db_name} not available")
            return None
        
        function_name = f"{operation}_vectors"
        func = getattr(module, function_name, None)
        
        if func is None:
            st.error(f"Function {function_name} not found in {db_name} module")
            return None
        
        # Call the function with provided kwargs and return the result
        result = func(**kwargs)
        return result
        
    except Exception as e:
        st.error(f"Error calling {db_name} {operation} function: {e}")
        return None

def create_comparison_charts(data: Dict, operation: str) -> List[Tuple[go.Figure, str, str]]:
    """Create comparison charts using line graphs instead of bar charts.
    Returns list of tuples (figure, chart_type, additional_info) for unique key generation."""
    figures = []
    
    if not data:
        return figures
    
    # Get all embedding models and sizes
    embedding_models = list(data.keys())
    all_sizes = set()
    all_databases = set()
    
    for model_data in data.values():
        all_sizes.update(model_data.keys())
        for size_data in model_data.values():
            all_databases.update(size_data.keys())
    
    all_sizes = sorted([int(s) for s in all_sizes])
    all_databases = sorted(list(all_databases))
    
    # Special handling for query operation which has two metrics
    if operation == "query":
        # Load similarity score data
        similarity_data = load_json_results("query_similarity")
        if similarity_data:
            # Create subplots for query throughput and similarity score
            fig1 = make_subplots(
                rows=len(embedding_models), 
                cols=2,
                subplot_titles=[f"{model} - Throughput" for model in embedding_models] + 
                             [f"{model} - Similarity Score" for model in embedding_models],
                shared_xaxes=True,
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Plot throughput data
            for i, model in enumerate(embedding_models, 1):
                for db in all_databases:
                    x_values = []
                    y_values = []
                    
                    for size in all_sizes:
                        size_str = str(size)
                        if size_str in data.get(model, {}) and db in data[model][size_str]:
                            x_values.append(size)
                            y_values.append(data[model][size_str][db])
                    
                    if x_values and y_values:
                        fig1.add_trace(
                            go.Scatter(
                                x=x_values,
                                y=y_values,
                                mode='lines+markers',
                                name=f"{db}",
                                line=dict(color=DATABASES.get(db, {}).get('color', '#333333'), width=3),
                                marker=dict(size=8),
                                legendgroup=db,
                                showlegend=(i == 1)
                            ),
                            row=i, col=1
                        )
            
            # Plot similarity score data
            for i, model in enumerate(embedding_models, 1):
                for db in all_databases:
                    x_values = []
                    y_values = []
                    
                    for size in all_sizes:
                        size_str = str(size)
                        if size_str in similarity_data.get(model, {}) and db in similarity_data[model][size_str]:
                            x_values.append(size)
                            y_values.append(similarity_data[model][size_str][db])
                    
                    if x_values and y_values:
                        fig1.add_trace(
                            go.Scatter(
                                x=x_values,
                                y=y_values,
                                mode='lines+markers',
                                name=f"{db}",
                                line=dict(color=DATABASES.get(db, {}).get('color', '#333333'), width=3),
                                marker=dict(size=8),
                                legendgroup=db,
                                showlegend=False
                            ),
                            row=i, col=2
                        )
            
            fig1.update_layout(
                height=300 * len(embedding_models),
                width=1200,
                title_text="Query Performance Metrics",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update y-axis labels
            for i in range(1, len(embedding_models) + 1):
                fig1.update_yaxes(title_text="Queries per second", row=i, col=1)
                fig1.update_yaxes(title_text="Similarity Score", row=i, col=2)
            
            figures.append((fig1, "performance", "query_metrics"))
        else:
            # Fallback to single metric if similarity data is not available
            fig1 = make_subplots(
                rows=len(embedding_models), 
                cols=1,
                subplot_titles=[f"{model}" for model in embedding_models],
                shared_xaxes=True,
                vertical_spacing=0.1
            )
            
            for i, model in enumerate(embedding_models, 1):
                for db in all_databases:
                    x_values = []
                    y_values = []
                    
                    for size in all_sizes:
                        size_str = str(size)
                        if size_str in data.get(model, {}) and db in data[model][size_str]:
                            x_values.append(size)
                            y_values.append(data[model][size_str][db])
                    
                    if x_values and y_values:
                        fig1.add_trace(
                            go.Scatter(
                                x=x_values,
                                y=y_values,
                                mode='lines+markers',
                                name=f"{db}",
                                line=dict(color=DATABASES.get(db, {}).get('color', '#333333'), width=3),
                                marker=dict(size=8),
                                legendgroup=db,
                                showlegend=(i == 1)
                            ),
                            row=i, col=1
                        )
            
            fig1.update_layout(
                height=300 * len(embedding_models),
                width=800,
                title_text="Query Throughput",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            figures.append((fig1, "performance", "query_throughput"))
    else:
        # Original chart creation for other operations
        fig1 = make_subplots(
            rows=len(embedding_models), 
            cols=1,
            subplot_titles=[f"{model}" for model in embedding_models],
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        for i, model in enumerate(embedding_models, 1):
            for db in all_databases:
                x_values = []
                y_values = []
                
                for size in all_sizes:
                    size_str = str(size)
                    if size_str in data.get(model, {}) and db in data[model][size_str]:
                        x_values.append(size)
                        y_values.append(data[model][size_str][db])
                
                if x_values and y_values:
                    fig1.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=y_values,
                            mode='lines+markers',
                            name=f"{db}",
                            line=dict(color=DATABASES.get(db, {}).get('color', '#333333'), width=3),
                            marker=dict(size=8),
                            legendgroup=db,
                            showlegend=(i == 1)
                        ),
                        row=i, col=1
                    )
        
        fig1.update_layout(
            height=300 * len(embedding_models),
            width=800,
            title_text=f"{operation.capitalize()} Performance",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        figures.append((fig1, "performance", operation))
    
    return figures

def create_trends_bar_charts(data: Dict, operation: str) -> List[go.Figure]:
    """Create interactive bar charts for trends visualization."""
    figures = []
    
    if not data:
        return figures
    
    # Get all available data dimensions
    embedding_models = list(data.keys())
    all_sizes = set()
    all_databases = set()
    
    for model_data in data.values():
        all_sizes.update(model_data.keys())
        for size_data in model_data.values():
            all_databases.update(size_data.keys())
    
    all_sizes = sorted([int(s) for s in all_sizes])
    all_databases = sorted(list(all_databases))
    
    # 1. Database Performance Comparison (for each embedding model and size)
    for model in embedding_models:
        for size in all_sizes:
            size_str = str(size)
            if size_str in data.get(model, {}):
                databases = []
                values = []
                colors = []
                
                for db in all_databases:
                    if db in data[model][size_str]:
                        databases.append(db.upper())
                        values.append(data[model][size_str][db])
                        colors.append(DATABASES.get(db, {}).get('color', '#333333'))
                
                if databases and values:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=databases,
                            y=values,
                            marker_color=colors,
                            text=[f"{v:.2f}" for v in values],
                            textposition='auto',
                            hovertemplate='<b>%{x}</b><br>Performance: %{y:.2f}<extra></extra>'
                        )
                    ])
                    
                    fig.update_layout(
                        title=f"Database Performance - {model} (Size: {size})",
                        xaxis_title="Database",
                        yaxis_title="Performance Value",
                        template="plotly_white",
                        height=400,
                        showlegend=False
                    )
                    
                    figures.append(fig)
    
    # 2. Size Scaling Performance (for each database and embedding model)
    for model in embedding_models:
        for db in all_databases:
            sizes = []
            values = []
            
            for size in all_sizes:
                size_str = str(size)
                if size_str in data.get(model, {}) and db in data[model][size_str]:
                    sizes.append(size)
                    values.append(data[model][size_str][db])
            
            if len(sizes) > 1 and values:  # Only create if we have multiple data points
                fig = go.Figure(data=[
                    go.Bar(
                        x=[str(s) for s in sizes],
                        y=values,
                        marker_color=DATABASES.get(db, {}).get('color', '#333333'),
                        text=[f"{v:.2f}" for v in values],
                        textposition='auto',
                        hovertemplate='<b>Size: %{x}</b><br>Performance: %{y:.2f}<extra></extra>'
                    )
                ])
                
                fig.update_layout(
                    title=f"Scaling Performance - {db.upper()} with {model}",
                    xaxis_title="Data Size",
                    yaxis_title="Performance Value",
                    template="plotly_white",
                    height=400,
                    showlegend=False
                )
                
                figures.append(fig)
    
    # 3. Embedding Model Comparison (for each database and size)
    for db in all_databases:
        for size in all_sizes:
            size_str = str(size)
            models = []
            values = []
            
            for model in embedding_models:
                if size_str in data.get(model, {}) and db in data[model][size_str]:
                    models.append(model)
                    values.append(data[model][size_str][db])
            
            if len(models) > 1 and values:  # Only create if we have multiple models
                fig = go.Figure(data=[
                    go.Bar(
                        x=models,
                        y=values,
                        marker_color=DATABASES.get(db, {}).get('color', '#333333'),
                        text=[f"{v:.2f}" for v in values],
                        textposition='auto',
                        hovertemplate='<b>%{x}</b><br>Performance: %{y:.2f}<extra></extra>'
                    )
                ])
                
                fig.update_layout(
                    title=f"Model Comparison - {db.upper()} (Size: {size})",
                    xaxis_title="Embedding Model",
                    yaxis_title="Performance Value",
                    template="plotly_white",
                    height=400,
                    showlegend=False
                )
                
                figures.append(fig)
    
    return figures

def create_results_table(data: Dict, operation: str) -> pd.DataFrame:
    """Create a comprehensive DataFrame with results for table display."""
    rows = []
    
    for embedding_model, model_data in data.items():
        for size, size_data in model_data.items():
            for db_name, value in size_data.items():
                row = {
                    'Embedding Model': embedding_model,
                    'Data Size': int(size),
                    'Database': db_name,
                    'Performance Value': value
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by embedding model, then size, then database
    if not df.empty:
        df = df.sort_values(['Embedding Model', 'Data Size', 'Database'])
    
    return df

def display_enhanced_trends_tab(data: Dict, operation: str):
    """Display enhanced trends tab with interactive filters and bar charts."""
    if not data:
        st.info("No data available for trends analysis. Please run benchmarks first.")
        return
    
    st.subheader("Performance Trends Analysis")
    st.markdown("Explore performance trends across different dimensions with interactive bar charts.")
    
    # Get available data dimensions
    embedding_models = list(data.keys())
    all_sizes = set()
    all_databases = set()
    
    for model_data in data.values():
        all_sizes.update(model_data.keys())
        for size_data in model_data.values():
            all_databases.update(size_data.keys())
    
    all_sizes = sorted([int(s) for s in all_sizes])
    all_databases = sorted(list(all_databases))
    
    # Create filter section
    st.markdown("### Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        view_type = st.selectbox(
            "Select View Type:",
            [
                "Database Comparison",
                "Size Scaling",
                "Model Comparison",
                "All Charts"
            ],
            key=f"view_type_{operation}_{st.session_state.chart_update_key}"
        )
    
    with filter_col2:
        if view_type in ["Database Comparison", "Size Scaling"]:
            selected_model = st.selectbox(
                "Select Embedding Model:",
                embedding_models,
                key=f"trend_model_{operation}_{st.session_state.chart_update_key}"
            )
        elif view_type == "Model Comparison":
            selected_db = st.selectbox(
                "Select Database:",
                all_databases,
                key=f"trend_db_{operation}_{st.session_state.chart_update_key}"
            )
    
    with filter_col3:
        if view_type == "Database Comparison":
            selected_size = st.selectbox(
                "Select Data Size:",
                all_sizes,
                key=f"trend_size_{operation}_{st.session_state.chart_update_key}"
            )
        elif view_type == "Model Comparison":
            selected_size = st.selectbox(
                "Select Data Size:",
                all_sizes,
                key=f"trend_size2_{operation}_{st.session_state.chart_update_key}"
            )
    
    st.markdown("---")
    
    # Display charts based on selected view type
    if view_type == "Database Comparison":
        st.markdown(f"### Database Performance Comparison - {selected_model} (Size: {selected_size})")
        
        size_str = str(selected_size)
        if size_str in data.get(selected_model, {}):
            databases = []
            values = []
            colors = []
            
            for db in all_databases:
                if db in data[selected_model][size_str]:
                    databases.append(db.upper())
                    values.append(data[selected_model][size_str][db])
                    colors.append(DATABASES.get(db, {}).get('color', '#333333'))
            
            if databases and values:
                fig = go.Figure(data=[
                    go.Bar(
                        x=databases,
                        y=values,
                        marker_color=colors,
                        text=[f"{v:.2f}" for v in values],
                        textposition='auto',
                        hovertemplate='<b>%{x}</b><br>Performance: %{y:.2f}<extra></extra>'
                    )
                ])
                
                fig.update_layout(
                    title=f"Database Performance Comparison",
                    xaxis_title="Database",
                    yaxis_title="Performance Value",
                    template="plotly_white",
                    height=500,
                    showlegend=False
                )
                
                chart_key = generate_chart_key(operation, "db_comparison", f"{selected_model}_{selected_size}")
                st.plotly_chart(fig, use_container_width=True, key=f"db_comp_{chart_key}")
                
                # Show top performers
                perf_data = list(zip(databases, values))
                perf_data.sort(key=lambda x: x[1], reverse=True)
                
                st.markdown("#### Top Performers")
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                
                if len(perf_data) >= 1:
                    with perf_col1:
                        st.metric("1st Place", perf_data[0][0], f"{perf_data[0][1]:.2f}")
                if len(perf_data) >= 2:
                    with perf_col2:
                        st.metric("2nd Place", perf_data[1][0], f"{perf_data[1][1]:.2f}")
                if len(perf_data) >= 3:
                    with perf_col3:
                        st.metric("3rd Place", perf_data[2][0], f"{perf_data[2][1]:.2f}")
            else:
                st.warning("No data available for the selected combination.")
    
    elif view_type == "Size Scaling":
        st.markdown(f"### Size Scaling Analysis - {selected_model}")
        
        # Create subplots for all databases
        fig = make_subplots(
            rows=(len(all_databases) + 1) // 2,
            cols=2,
            subplot_titles=[f"{db.upper()}" for db in all_databases],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for idx, db in enumerate(all_databases):
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            sizes = []
            values = []
            
            for size in all_sizes:
                size_str = str(size)
                if size_str in data.get(selected_model, {}) and db in data[selected_model][size_str]:
                    sizes.append(size)
                    values.append(data[selected_model][size_str][db])
            
            if sizes and values:
                fig.add_trace(
                    go.Bar(
                        x=[str(s) for s in sizes],
                        y=values,
                        marker_color=DATABASES.get(db, {}).get('color', '#333333'),
                        text=[f"{v:.2f}" for v in values],
                        textposition='auto',
                        name=db.upper(),
                        showlegend=False,
                        hovertemplate=f'<b>{db.upper()}</b><br>Size: %{{x}}<br>Performance: %{{y:.2f}}<extra></extra>'
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=300 * ((len(all_databases) + 1) // 2),
            title_text=f"Size Scaling Performance - {selected_model}",
            template="plotly_white"
        )
        
        # Update axis labels
        for i in range(1, ((len(all_databases) + 1) // 2) + 1):
            fig.update_xaxes(title_text="Data Size", row=i, col=1)
            fig.update_xaxes(title_text="Data Size", row=i, col=2)
            fig.update_yaxes(title_text="Performance", row=i, col=1)
            fig.update_yaxes(title_text="Performance", row=i, col=2)
        
        chart_key = generate_chart_key(operation, "size_scaling", selected_model)
        st.plotly_chart(fig, use_container_width=True, key=f"size_scale_{chart_key}")
    
    elif view_type == "Model Comparison":
        st.markdown(f"### Model Comparison - {selected_db.upper()} (Size: {selected_size})")
        
        size_str = str(selected_size)
        models = []
        values = []
        
        for model in embedding_models:
            if size_str in data.get(model, {}) and selected_db in data[model][size_str]:
                models.append(model)
                values.append(data[model][size_str][selected_db])
        
        if models and values:
            fig = go.Figure(data=[
                go.Bar(
                    x=models,
                    y=values,
                    marker_color=DATABASES.get(selected_db, {}).get('color', '#333333'),
                    text=[f"{v:.2f}" for v in values],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Performance: %{y:.2f}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title=f"Model Performance Comparison - {selected_db.upper()}",
                xaxis_title="Embedding Model",
                yaxis_title="Performance Value",
                template="plotly_white",
                height=500,
                showlegend=False
            )
            
            chart_key = generate_chart_key(operation, "model_comparison", f"{selected_db}_{selected_size}")
            st.plotly_chart(fig, use_container_width=True, key=f"model_comp_{chart_key}")
            
            # Show model performance insights
            if len(values) > 1:
                best_idx = values.index(max(values))
                worst_idx = values.index(min(values))
                
                insight_col1, insight_col2 = st.columns(2)
                with insight_col1:
                    st.success(f"**Best Model**: {models[best_idx]} ({values[best_idx]:.2f})")
                with insight_col2:
                    if best_idx != worst_idx:
                        improvement = ((values[best_idx] - values[worst_idx]) / values[worst_idx]) * 100
                        st.info(f" **Improvement**: {improvement:.1f}% over lowest performer")
        else:
            st.warning("No data available for the selected combination.")
    
    elif view_type == "All Charts":
        st.markdown("### Complete Trends Overview")
        
        # Create all possible charts
        all_figures = create_trends_bar_charts(data, operation)
        
        if all_figures:
            # Group charts by type
            chart_types = {
                "Database Comparisons": [],
                "Size Scaling": [],
                "Model Comparisons": []
            }
            
            for fig in all_figures:
                title = fig.layout.title.text
                if "Database Performance" in title:
                    chart_types["Database Comparisons"].append(fig)
                elif "Scaling Performance" in title:
                    chart_types["Size Scaling"].append(fig)
                elif "Model Comparison" in title:
                    chart_types["Model Comparisons"].append(fig)
            
            # Display charts in tabs
            if any(chart_types.values()):
                tab1, tab2, tab3 = st.tabs(["Database Comparisons", "Size Scaling", "Model Comparisons"])
                
                with tab1:
                    if chart_types["Database Comparisons"]:
                        for i, fig in enumerate(chart_types["Database Comparisons"]):
                            chart_key = generate_chart_key(operation, "all_db", str(i))
                            st.plotly_chart(fig, use_container_width=True, key=f"all_db_{chart_key}")
                    else:
                        st.info("No database comparison charts available.")
                
                with tab2:
                    if chart_types["Size Scaling"]:
                        for i, fig in enumerate(chart_types["Size Scaling"]):
                            chart_key = generate_chart_key(operation, "all_size", str(i))
                            st.plotly_chart(fig, use_container_width=True, key=f"all_size_{chart_key}")
                    else:
                        st.info("No size scaling charts available.")
                
                with tab3:
                    if chart_types["Model Comparisons"]:
                        for i, fig in enumerate(chart_types["Model Comparisons"]):
                            chart_key = generate_chart_key(operation, "all_model", str(i))
                            st.plotly_chart(fig, use_container_width=True, key=f"all_model_{chart_key}")
                    else:
                        st.info("No model comparison charts available.")
        else:
            st.info("No charts available. Please run benchmarks first.")

def display_results(operation: str):
    """Display benchmark results with enhanced charts and tables."""
    data = load_json_results(operation)
    
    if not data:
        st.warning(f"No results found for {operation} operation. Please run the benchmark first.")
        return
    
    st.subheader(f"{operation.title()} Operation Results")
    
    # Create tabs for different views with shorter names
    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Trends", "Results", "Explorer"])
    
    with tab1:
        figures = create_comparison_charts(data, operation)
        if figures:
            fig, chart_type, additional_info = figures[0]
            chart_key = generate_chart_key(operation, chart_type, additional_info)
            st.plotly_chart(fig, use_container_width=True, key=f"main_chart_{chart_key}")
    
    with tab2:
        # Use the new enhanced trends display
        display_enhanced_trends_tab(data, operation)
    
    with tab3:
        df = create_results_table(data, operation)
        if not df.empty:
            st.dataframe(df, use_container_width=True, key=f"results_table_{operation}_{st.session_state.chart_update_key}")
            
            # Download button for CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"{operation}_results.csv",
                mime="text/csv",
                key=f"download_{operation}_{st.session_state.chart_update_key}"
            )
        else:
            st.info("No detailed results available.")
    
    with tab4:
        if not data:
            st.info("No data to explore.")
            return
            
        st.subheader("Interactive Data Explorer")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            available_models = list(data.keys())
            selected_model = st.selectbox(
                "Select Embedding Model:", 
                available_models, 
                key=f"model_selector_{operation}_{st.session_state.chart_update_key}"
            )
        
        with col2:
            if selected_model in data:
                available_sizes = list(data[selected_model].keys())
                selected_size = st.selectbox(
                    "Select Data Size:", 
                    available_sizes, 
                    key=f"size_selector_{operation}_{st.session_state.chart_update_key}"
                )
        
        # Display filtered results
        if selected_model in data and selected_size in data[selected_model]:
            filtered_data = data[selected_model][selected_size]
            
            # Create a simple bar chart for the selected combination
            databases = list(filtered_data.keys())
            values = list(filtered_data.values())
            colors = [DATABASES.get(db, {}).get('color', '#333333') for db in databases]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=databases,
                    y=values,
                    marker_color=colors,
                    text=values,
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f"{operation.title()} Performance - {selected_model} (Size: {selected_size})",
                xaxis_title="Database",
                yaxis_title="Performance Value",
                template="plotly_white",
                height=400
            )
            
            chart_key = generate_chart_key(operation, "explorer", f"{selected_model}_{selected_size}")
            st.plotly_chart(fig, use_container_width=True, key=f"explorer_chart_{chart_key}")
            
            # Show raw data
            st.subheader("Raw Data")
            filtered_df = pd.DataFrame(list(filtered_data.items()), columns=['Database', 'Value'])
            st.dataframe(filtered_df, use_container_width=True, key=f"explorer_table_{chart_key}")

@st.cache_data
def get_results_hash(operation: str) -> str:
    """Get a hash of the current results to detect changes."""
    data = load_json_results(operation)
    if data:
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return ""

def main():
    st.title("Vector Database Benchmarking")
    st.markdown("Compare the performance of different vector databases across multiple embedding models and data sizes")

    if import_errors:
        with st.expander("Import Warnings", expanded=False):
            for error in import_errors:
                st.warning(error)

    st.sidebar.header("Databases")
    available_databases = get_available_databases()
    if not available_databases:
        st.sidebar.error("No database modules are available. Please check your imports.")
        st.stop()
    
    # Add "Select All" checkbox
    select_all = st.sidebar.checkbox("Select All", key="select_all")
    
    # Database checkboxes
    selected_databases = [
        db for db in available_databases
        if select_all or st.sidebar.checkbox(db.upper(), key=f"db_{db}")
    ]
    
    # Show unavailable databases
    unavailable_databases = [db for db in DATABASES if db not in available_databases]
    if unavailable_databases:
        st.sidebar.info(f"Unavailable: {', '.join(unavailable_databases)}")

    st.sidebar.header("Embedding Model")
    selected_model = st.sidebar.selectbox("Select an embedding model", EMBEDDING_MODELS)
    selected_embedding_models = [selected_model]  # Keeps the datatype as a list
            

    available_datasets = get_available_datasets(selected_model)
    selected_dataset = st.sidebar.selectbox("Choose dataset:", available_datasets)

    selected_sizes = [s for s in DEFAULT_SIZES if st.sidebar.checkbox(f"{s} rows", key=f"size_{s}")]
    custom_size = st.sidebar.number_input("Custom size:", min_value=100, max_value=100000, value=1000, step=100)
    if st.sidebar.checkbox("Use custom size", key="custom_size_check") and custom_size not in selected_sizes:
        selected_sizes.append(custom_size)
    if not selected_sizes:
        st.sidebar.warning("Please select at least one data size.")

    operation = st.sidebar.selectbox("Choose operation:", OPERATIONS)
    
    # Add query input section for query operation
    query_text = None
    k_results = 5
    if operation == 'query':
        st.sidebar.markdown("---")
        st.sidebar.subheader("Query Parameters")
        query_text = st.sidebar.text_input("Enter your query text", "Protocol with TCP")
        k_results = st.sidebar.number_input("Number of results to return", min_value=1, max_value=100, value=5)
    
    # Update chart keys when operation changes
    if st.session_state.last_operation != operation:
        st.session_state.last_operation = operation
        update_chart_keys()
    
    faiss_index_type = None
    if 'faiss' in selected_databases:
        st.sidebar.subheader("6. FAISS Configuration")
        faiss_index_type = st.sidebar.selectbox("FAISS Index Type:", FAISS_INDEX_TYPES)
        if faiss_index_type == 'HNSW' and operation == 'delete':
            st.sidebar.error("FAISS HNSW does not support delete operations!")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Benchmark Configuration")
        config = {
            "Selected Databases": ", ".join(selected_databases) or "None",
            "Embedding Models": ", ".join(selected_embedding_models) or "None",
            "Dataset": selected_dataset,
            "Data Sizes": ", ".join(map(str, sorted(selected_sizes))) or "None",
            "Operation": operation.title()
        }
        if faiss_index_type and 'faiss' in selected_databases:
            config["FAISS Index Type"] = faiss_index_type
        st.table(pd.DataFrame(list(config.items()), columns=["Parameter", "Value"]))

    with col2:
        st.subheader("Actions")
        is_valid, error_msg = validate_inputs(selected_databases, operation, faiss_index_type or "", selected_sizes)
        if not selected_embedding_models:
            is_valid = False
            error_msg = "Please select at least one embedding model."
        if not is_valid:
            st.error(error_msg)
            run_disabled = True
        else:
            run_disabled = False

        if st.button("Run Benchmark", disabled=run_disabled, type="primary"):
            # Update chart keys before running benchmark
            update_chart_keys()
            with st.spinner("Running comprehensive benchmark..."):
                total = len(selected_databases) * len(selected_embedding_models) * len(selected_sizes)
                progress = st.progress(0)
                count = 0
                for model in selected_embedding_models:
                    for size in selected_sizes:
                        for db_name in selected_databases:
                            count += 1
                            st.info(f"Running {operation} for {db_name} with {model} (size: {size}) [{count}/{total}]")
                            kwargs = {'embedding_model': model, 'dataset': selected_dataset, 'nrows': size}
                            if db_name == 'faiss' and faiss_index_type:
                                kwargs['index_type'] = faiss_index_type
                            if operation == 'query':
                                kwargs['query_text'] = query_text
                                kwargs['k'] = k_results
                            result = call_database_function(db_name, operation, **kwargs)
                            if result is not None:
                                if operation == 'query':
                                    throughput, similarity, results = result
                                    # Display query results in col1
                                    with col1:
                                        st.subheader(f"Query Results for {db_name} with {model} (size: {size})")
                                        if results:
                                            results_df = pd.DataFrame(results)
                                            st.dataframe(results_df)
                                        else:
                                            st.info("No results found")
                                    # Display metrics outside of any column context
                                    col_metric1, col_metric2 = st.columns(2)
                                    with col_metric1:
                                        st.metric("Query Throughput", f"{throughput:.2f} queries/sec")
                                    with col_metric2:
                                        st.metric("Average Similarity", f"{similarity:.2f}")
                                    value = throughput  # Use throughput for the results structure
                                else:
                                    value = result[0] if isinstance(result, (list, tuple)) else result if isinstance(result, (int, float)) else 0
                                update_results_structure(operation, model, size, db_name, value)
                            progress.progress(count / total)
                st.session_state.benchmark_completed = True
                # Update chart keys after completing benchmark
                update_chart_keys()
                st.success("Benchmark completed!")
                # Display results immediately after completion
                with col1:
                    display_results(operation)

    # Always show existing results below (but with updated keys)
    if not st.session_state.benchmark_completed:
        display_results(operation)

if __name__ == "__main__":
    main()