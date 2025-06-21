import pandas as pd
import json
import os
from typing import Dict, List, Optional, Tuple
import hashlib
import time
import logging


# Configure logging
logging.basicConfig(
    filename='master.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import database modules
import_errors = []


try:
    import faiss_module as faiss_db
except ImportError as e:
    error_msg = f"faiss_module: {e}"
    import_errors.append(error_msg)
    logger.error(error_msg)
    faiss_db = None

try:
    import milvus_module as milvus_db
except ImportError as e:
    error_msg = f"milvus_module: {e}"
    import_errors.append(error_msg)
    logger.error(error_msg)
    milvus_db = None

try:
    import chroma_module as chroma_db
except ImportError as e:
    error_msg = f"chroma_module: {e}"
    import_errors.append(error_msg)
    logger.error(error_msg)
    chroma_db = None

try:
    import weaviate_module as weaviate_db
except ImportError as e:
    error_msg = f"weaviate_module: {e}"
    import_errors.append(error_msg)
    logger.error(error_msg)
    weaviate_db = None

try:
    import qdrant_module as qdrant_db
except ImportError as e:
    error_msg = f"qdrant_module: {e}"
    import_errors.append(error_msg)
    logger.error(error_msg)
    qdrant_db = None

try:
    import pgvector_module as pgvector_db
except ImportError as e:
    error_msg = f"pgvector_module: {e}"
    import_errors.append(error_msg)
    logger.error(error_msg)
    pgvector_db = None

# Database configuration
DATABASES = {
    'faiss': {'module': faiss_db, 'color': '#FF6B6B'},
    'milvus': {'module': milvus_db, 'color': '#4ECDC4'},
    'chroma': {'module': chroma_db, 'color': '#45B7D1'},
    'weaviate': {'module': weaviate_db, 'color': '#96CEB4'},
    'qdrant': {'module': qdrant_db, 'color': '#FFEAA7'},
    'pgvector': {'module': pgvector_db, 'color': '#DDA0DD'}
}

OPERATIONS = ['create', 'query', 'delete']
EMBEDDING_MODELS = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'snowflake-arctic-embed-l-v2.0']
FAISS_INDEX_TYPES = ['IVFFLAT', 'HNSW']
DEFAULT_SIZES = [500, 1000, 2000, 5000]

def load_json_results(operation: str) -> Optional[Dict]:
    """Load benchmark results from JSON file with new nested structure."""
    filename = f"{operation}.json"
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                message = f"Successfully loaded results from {filename}"
                #print(message)
                logger.info(message)
                return data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            message = f"Error loading {filename}: {e}"
            print(message)
            logger.error(message)
            return None
    else:
        message = f"File {filename} does not exist."
        print(message)
        logger.warning(message)
        return None

def save_json_results(operation: str, data: Dict):
    """Save benchmark results to JSON file with new nested structure."""
    filename = f"{operation}.json"
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        message = f"Results successfully saved to {filename}"
        #print(message)
        logger.info(message)
    except Exception as e:
        message = f"Error saving {filename}: {e}"
        print(message)
        logger.error(message)

def update_results_structure(operation: str, embedding_model: str, size: int, db_name: str, value: float):
    """Update the results JSON with the new nested structure."""
    existing_data = load_json_results(operation) or {}
    
    if embedding_model not in existing_data:
        existing_data[embedding_model] = {}
    
    if str(size) not in existing_data[embedding_model]:
        existing_data[embedding_model][str(size)] = {}
    
    existing_data[embedding_model][str(size)][db_name] = value
    save_json_results(operation, existing_data)

def get_available_datasets() -> List[str]:
    """Get list of available CSV datasets that match the embedding model name."""
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    matching_datasets = []
    for csv_file in csv_files:
        if any(model in csv_file for model in EMBEDDING_MODELS):
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
            message = f"Module for {db_name} not available"
            print(message)
            logger.warning(message)
            return None
        
        function_name = f"{operation}_vectors"
        func = getattr(module, function_name, None)
        
        if func is None:
            message = f"Function {function_name} not found in {db_name} module"
            print(message)
            logger.warning(message)
            return None
        
        result = func(**kwargs)
        return result

    except Exception as e:
        message = f"Error calling {db_name} {operation} function: {e}"
        print(message)
        logger.error(message)
        return None

def run_benchmark(
    selected_databases: List[str],
    selected_embedding_models: List[str],
    selected_dataset: str,
    selected_sizes: List[int],
    operation: str,
    faiss_index_type: Optional[str] = None,
    query_text: Optional[str] = None,
    k_results: int = 5
) -> Dict:
    """
    Run the benchmark with the given parameters and return the results.
    
    Args:
        selected_databases: List of database names to benchmark
        selected_embedding_models: List of embedding models to use
        selected_dataset: Name of the dataset to use
        selected_sizes: List of data sizes to test
        operation: Operation to benchmark ('create', 'query', or 'delete')
        faiss_index_type: Optional FAISS index type if using FAISS
        query_text: Optional query text for query operations
        k_results: Number of results to return for query operations
        
    Returns:
        Dict containing the benchmark results
    """
    results = {}
    
    # Validate inputs
    is_valid, error_msg = validate_inputs(selected_databases, operation, faiss_index_type or "", selected_sizes)
    if not is_valid:
        raise ValueError(error_msg)
    
    if not selected_embedding_models:
        raise ValueError("Please select at least one embedding model.")
    
    total = len(selected_databases) * len(selected_embedding_models) * len(selected_sizes)
    count = 0
    
    for model in selected_embedding_models:
        results[model] = {}
        for size in selected_sizes:
            results[model][str(size)] = {}
            for db_name in selected_databases:
                count += 1
                message = f"Running {operation} for {db_name} with {model} (size: {size}) [{count}/{total}]"
                print(message)
                logger.info(message)
                
                kwargs = {'embedding_model': model, 'dataset': selected_dataset, 'nrows': size}
                if db_name == 'faiss' and faiss_index_type:
                    kwargs['index_type'] = faiss_index_type
                if operation == 'query':
                    kwargs['query_text'] = query_text
                    kwargs['k'] = k_results
                
                result = call_database_function(db_name, operation, **kwargs)
                
                if result is not None:
                    if operation == 'query':
                        throughput, similarity, _ = result
                        value = throughput
                        results[model][str(size)][db_name] = {
                            'throughput': throughput,
                            'similarity': similarity
                        }
                    else:
                        value = result[0] if isinstance(result, (list, tuple)) else result if isinstance(result, (int, float)) else 0
                        results[model][str(size)][db_name] = value
                    
                    update_results_structure(operation, model, size, db_name, value)
    
    return results

def get_results_hash(operation: str) -> str:
    """Get a hash of the current results to detect changes."""
    data = load_json_results(operation)
    if data:
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
