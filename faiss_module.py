import os
import time
import json
import ast
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import random

_faiss_storage = {}

# Helper to load a subset of the dataset
def load_dataset(dataset, nrows):
    df = pd.read_csv(dataset)
    if nrows is not None and nrows < len(df):
        df = df.iloc[:nrows]
    return df

# Helper to parse vector string to list of floats
def parse_vector(vector_str):
    try:
        # Remove any extra whitespace and parse as Python literal
        vector_list = ast.literal_eval(vector_str.strip())
        return vector_list
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing vector: {e}")
        return None

def process_csv_data(df):
    vectors = []
    texts = []
    indices = []
    
    for _, row in df.iterrows():
        # Parse the precomputed vector
        vector = parse_vector(row["vector"])
        if vector is None:
            print(f"Skipping row {row['index']} due to vector parsing error")
            continue
        
        vectors.append(vector)
        texts.append(row["metadata"])
        indices.append(int(row["index"]))
    
    return np.array(vectors, dtype=np.float32), texts, indices

def create_faiss_index(embeddings, index_type='IVFFLAT'):
    dimension = embeddings.shape[1]
    n_vectors = len(embeddings)
    
    if index_type == 'HNSW':
        # HNSW index
        index = faiss.IndexHNSWFlat(dimension)
        index.hnsw.M = 16
        index.hnsw.efConstruction = 200
        index.add(embeddings)
    else:  # IVFFLAT
        # IVF index for better performance
        nlist = min(100, max(1, int(np.sqrt(n_vectors))))
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        
        # Train and add data
        index.train(embeddings)
        index.add(embeddings)
    
    return index

def get_storage_key(embedding_model, dataset, nrows, index_type=None):
    key = f"{embedding_model}_{dataset}_{nrows}"
    if index_type:
        key += f"_{index_type}"
    return key

# CREATE operation: ingest vectors and return index construction time (to last object searchable)
def create_vectors(embedding_model, dataset, nrows, index_type='IVFFLAT'):
    df = load_dataset(dataset, nrows)
    
    start_time = time.time()
    
    # Process data to get precomputed vectors and metadata
    embeddings, data_texts, data_indices = process_csv_data(df)
    
    # Normalize embeddings properly
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Create FAISS index
    index = create_faiss_index(embeddings, index_type)
    
    total_time = time.time() - start_time
    
    # Calculate throughput (vectors per second)
    throughput = len(embeddings) / total_time if total_time > 0 else 0
    
    # Store for later operations
    storage_key = get_storage_key(embedding_model, dataset, nrows, index_type)
    _faiss_storage[storage_key] = {
        'index': index,
        'embeddings': embeddings,
        'texts': data_texts,
        'indices': data_indices,
        'index_type': index_type
    }
    
    # Save result to create.json (as master.py expects)
    save_result("create", embedding_model, nrows, "faiss", throughput)
    return throughput

# QUERY operation: run a semantic query and return average latency
def query_vectors(embedding_model, dataset, nrows, query_text=None, k=5, index_type='IVFFLAT'):
    storage_key = get_storage_key(embedding_model, dataset, nrows, index_type)
    
    # Check if we have stored data, if not create it
    if storage_key not in _faiss_storage:
        create_vectors(embedding_model, dataset, nrows, index_type)
    
    stored_data = _faiss_storage[storage_key]
    index = stored_data['index']
    texts = stored_data['texts']
    indices = stored_data['indices']

    # Map model alias to Hugging Face model ID
    if embedding_model == "snowflake-arctic-embed-l-v2.0":
        hf_model_name = "Snowflake/snowflake-arctic-embed-l-v2.0"
    else:
        hf_model_name = embedding_model

    model = SentenceTransformer(hf_model_name)
    
    # If no query text provided, use a default
    if query_text is None:
        query_text = "Protocol with TCP"
    
    # Encode the query text and normalize it
    vector = model.encode(query_text)
    query_embedding = np.array([vector], dtype=np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    # Set nprobe for IVF index
    if hasattr(index, 'nprobe'):
        index.nprobe = min(10, getattr(index, 'nlist', 10))
    
    # Ensure k doesn't exceed available vectors
    k = min(k, index.ntotal, len(texts))
    
    start = time.time()
    
    # Search (single query like Weaviate)
    distances, faiss_indices = index.search(query_embedding, k)
    latency = time.time() - start
    
    # Process results
    results = []
    if len(distances[0]) > 0:
        for i, (dist, idx) in enumerate(zip(distances[0], faiss_indices[0])):
            # Check bounds to prevent index errors
            if idx >= 0 and idx < len(texts) and idx < len(indices):
                # Convert L2 distance to cosine similarity
                # For normalized vectors, L2 distance = 2 * (1 - cosine_similarity)
                # So cosine_similarity = 1 - (L2_distance / 2)
                similarity = max(0, 1 - (dist / 2))  # Clamp to avoid negative values
                results.append({
                    "flow_index": indices[idx],
                    "metadata": texts[idx],
                    "similarity": similarity
                })
            else:
                print(f"Warning: Invalid index {idx} (max: {len(texts)-1}) or distance {dist}")
    
    # Calculate metrics
    query_throughput = 1 / latency if latency > 0 else 0
    avg_similarity = sum(r["similarity"] for r in results) / len(results) if results else 0
    
    # Save metrics
    save_result("query_throughput", embedding_model, nrows, "faiss", query_throughput)
    save_result("query_similarity", embedding_model, nrows, "faiss", avg_similarity)
    
    return query_throughput, avg_similarity, results

# DELETE operation: delete all objects and return time taken
def delete_vectors(embedding_model, dataset, nrows, index_type='IVFFLAT'):
    storage_key = get_storage_key(embedding_model, dataset, nrows, index_type)
    
    # Check if we have stored data, if not create it
    if storage_key not in _faiss_storage:
        create_vectors(embedding_model, dataset, nrows, index_type)
    
    # Check if HNSW index (doesn't support deletion)
    if index_type == 'HNSW':
        # For HNSW, we simulate deletion by clearing the storage
        start = time.time()
        stored_data = _faiss_storage[storage_key]
        total_vectors = stored_data['index'].ntotal
        num_to_delete = min(nrows, total_vectors)
        
        # Clear the storage to simulate deletion
        del _faiss_storage[storage_key]
        
        duration = time.time() - start
        delete_throughput = num_to_delete / duration if duration > 0 else 0
        save_result("delete", embedding_model, nrows, "faiss", delete_throughput)
        return delete_throughput
    
    stored_data = _faiss_storage[storage_key]
    index = stored_data['index']
    embeddings = stored_data['embeddings']
    texts = stored_data['texts']
    indices = stored_data['indices']
    
    start = time.time()
    
    # Get total available vectors
    total_vectors = len(embeddings)  # Use embeddings length instead of index.ntotal
    
    # Calculate number to delete
    num_to_delete = min(nrows, total_vectors)
    
    if num_to_delete > 0 and total_vectors > 0:
        # Select random indices to delete
        indices_to_delete = set(random.sample(range(total_vectors), num_to_delete))
        indices_to_keep = [i for i in range(total_vectors) if i not in indices_to_delete]
        
        if indices_to_keep:
            # Create new arrays with remaining vectors
            kept_embeddings = embeddings[indices_to_keep]
            kept_texts = [texts[i] for i in indices_to_keep]
            kept_indices = [indices[i] for i in indices_to_keep]
            
            # Create new index with remaining vectors
            new_index = create_faiss_index(kept_embeddings, index_type)
            
            # Update stored data
            stored_data['index'] = new_index
            stored_data['embeddings'] = kept_embeddings
            stored_data['texts'] = kept_texts
            stored_data['indices'] = kept_indices
        else:
            # Delete all vectors - clear the storage
            del _faiss_storage[storage_key]
        
        total_deleted = num_to_delete
    else:
        total_deleted = 0
    
    duration = time.time() - start
    
    # Calculate delete throughput (deletions per second)
    # Add minimum duration to prevent extremely high values
    duration = max(duration, 0.001)  # Minimum 1ms
    delete_throughput = total_deleted / duration
    
    save_result("delete", embedding_model, nrows, "faiss", delete_throughput)
    return delete_throughput

# Save result to JSON file in the nested structure expected by master.py
def save_result(operation, embedding_model, nrows, db_name, value):
    filename = f"{operation}.json"
    
    # Ensure the value is JSON serializable
    if isinstance(value, (np.int32, np.int64)):
        value = int(value)
    elif isinstance(value, (np.float32, np.float64)):
        value = float(value)
    
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        
        if embedding_model not in data:
            data[embedding_model] = {}
        if str(nrows) not in data[embedding_model]:
            data[embedding_model][str(nrows)] = {}
        
        data[embedding_model][str(nrows)][db_name] = value
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        print(f"Error saving result to {filename}: {e}")
        print(f"Attempted to save: {operation}, {embedding_model}, {nrows}, {db_name}, {value}")
        raise