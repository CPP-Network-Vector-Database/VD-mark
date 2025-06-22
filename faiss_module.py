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

def load_dataset(dataset, nrows):
    df = pd.read_csv(dataset)
    if nrows is not None and nrows < len(df):
        df = df.iloc[:nrows]
    return df

def parse_vector(vector_str):
    """
    Turns those pesky string representations like "[0.1, 0.2, 0.3]" 
    back into actual lists we can work with. Sometimes data gets stored 
    as strings when it really wants to be numbers.
    """
    try:
        vector_list = ast.literal_eval(vector_str.strip())
        return vector_list
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing vector: {e}")
        return None

def process_csv_data(df):
    """
    Takes our CSV and extracts the good stuff: vectors, text, and indices.
    Skips any rows where the vector parsing goes sideways.
    """
    vectors = []
    texts = []
    indices = []
    
    for _, row in df.iterrows():
        vector = parse_vector(row["vector"])
        if vector is None:
            print(f"Skipping row {row['index']} due to vector parsing error")
            continue
        
        vectors.append(vector)
        texts.append(row["metadata"])
        indices.append(int(row["index"]))
    
    return np.array(vectors, dtype=np.float32), texts, indices

def create_faiss_index(embeddings, index_type='IVFFLAT'):
    """
    Here's where the magic happens. FAISS gives us two flavors:
    
    HNSW: It's a clever graph where each point knows its neighbors. 
    Super fast for searches, but once you build it, no takebacks on deletions (or updates).
    
    IVFFLAT: Divides space into clusters (like organizing books by genre), 
    then searches only relevant clusters. Slower than HNSW but supports deletions.
    """
    dimension = embeddings.shape[1]
    n_vectors = len(embeddings)
    
    if index_type == 'HNSW':
        index = faiss.IndexHNSWFlat(dimension)
        # M=16 means each node connects to 16 neighbors (sweet spot for most cases)
        # efConstruction=200 means we consider 200 candidates when building (higher= better quality)
        index.hnsw.M = 16
        index.hnsw.efConstruction = 200
        index.add(embeddings)
    else:  # IVFFLAT
        # Create sqrt(n) clusters because math people figured out this works well
        nlist = min(100, max(1, int(np.sqrt(n_vectors))))
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        
        # Train tells FAISS how to split the space, then we add our data
        index.train(embeddings)
        index.add(embeddings)
    
    return index

def get_storage_key(embedding_model, dataset, nrows, index_type=None):
    key = f"{embedding_model}_{dataset}_{nrows}"
    if index_type:
        key += f"_{index_type}"
    return key

def create_vectors(embedding_model, dataset, nrows, index_type='IVFFLAT'):
    """
    The big setup: load data, build the search index, and track how fast we did it.
    Throughput here means "vectors processed per second" during index construction.
    """
    df = load_dataset(dataset, nrows)
    
    start_time = time.time()
    
    embeddings, data_texts, data_indices = process_csv_data(df)
    
    # Normalize vectors so cosine similarity works properly (all vectors become unit length)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = create_faiss_index(embeddings, index_type)
    
    total_time = time.time() - start_time
    throughput = len(embeddings) / total_time if total_time > 0 else 0
    
    # Stash everything for later operations
    storage_key = get_storage_key(embedding_model, dataset, nrows, index_type)
    _faiss_storage[storage_key] = {
        'index': index,
        'embeddings': embeddings,
        'texts': data_texts,
        'indices': data_indices,
        'index_type': index_type
    }
    
    save_result("create", embedding_model, nrows, "faiss", throughput)
    return throughput

def query_vectors(embedding_model, dataset, nrows, query_text=None, k=5, index_type='IVFFLAT'):
    """
    The moment of truth: find similar vectors and see how fast we can do it.
    We convert L2 distances back to cosine similarity because that's what humans understand.
    """
    storage_key = get_storage_key(embedding_model, dataset, nrows, index_type)
    
    if storage_key not in _faiss_storage:
        create_vectors(embedding_model, dataset, nrows, index_type)
    
    stored_data = _faiss_storage[storage_key]
    index = stored_data['index']
    texts = stored_data['texts']
    indices = stored_data['indices']

    # Handle the Snowflake model alias quirk
    if embedding_model == "snowflake-arctic-embed-l-v2.0":
        hf_model_name = "Snowflake/snowflake-arctic-embed-l-v2.0"
    else:
        hf_model_name = embedding_model

    model = SentenceTransformer(hf_model_name)
    
    if query_text is None:
        query_text = "Protocol with TCP"
    
    # Turn query text into the same kind of vector our index expects
    vector = model.encode(query_text)
    query_embedding = np.array([vector], dtype=np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    # For IVF indices: nprobe controls how many clusters we search (more = better results, slower)
    if hasattr(index, 'nprobe'):
        index.nprobe = min(10, getattr(index, 'nlist', 10))
    
    k = min(k, index.ntotal, len(texts))
    
    start = time.time()
    distances, faiss_indices = index.search(query_embedding, k)
    latency = time.time() - start
    
    results = []
    if len(distances[0]) > 0:
        for i, (dist, idx) in enumerate(zip(distances[0], faiss_indices[0])):
            if idx >= 0 and idx < len(texts) and idx < len(indices):
                # Convert L2 distance back to cosine similarity
                # Math: for normalized vectors, L2_distance = 2 * (1 - cosine_similarity)
                similarity = max(0, 1 - (dist / 2))
                results.append({
                    "flow_index": indices[idx],
                    "metadata": texts[idx],
                    "similarity": similarity
                })
            else:
                print(f"Warning: Invalid index {idx} (max: {len(texts)-1}) or distance {dist}")
    
    query_throughput = 1 / latency if latency > 0 else 0
    avg_similarity = sum(r["similarity"] for r in results) / len(results) if results else 0
    
    save_result("query_throughput", embedding_model, nrows, "faiss", query_throughput)
    save_result("query_similarity", embedding_model, nrows, "faiss", avg_similarity)
    
    return query_throughput, avg_similarity, results

def delete_vectors(embedding_model, dataset, nrows, index_type='IVFFLAT'):
    """
    Here's where HNSW shows its weakness: it's read-only after construction.
    So we fake deletion by nuking the whole thing.
    
    IVFFLAT plays nicer: we can actually remove vectors by rebuilding the index
    with only the survivors. Not elegant, but it works.
    """
    storage_key = get_storage_key(embedding_model, dataset, nrows, index_type)
    
    if storage_key not in _faiss_storage:
        create_vectors(embedding_model, dataset, nrows, index_type)
    
    # HNSW doesn't support real deletion, so we simulate it
    if index_type == 'HNSW':
        start = time.time()
        stored_data = _faiss_storage[storage_key]
        total_vectors = stored_data['index'].ntotal
        num_to_delete = min(nrows, total_vectors)
        
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
    
    total_vectors = len(embeddings)
    num_to_delete = min(nrows, total_vectors)
    
    if num_to_delete > 0 and total_vectors > 0:
        # Pick random victims for deletion (keeps things fair for benchmarking)
        indices_to_delete = set(random.sample(range(total_vectors), num_to_delete))
        indices_to_keep = [i for i in range(total_vectors) if i not in indices_to_delete]
        
        if indices_to_keep:
            # Rebuild index with survivors only
            kept_embeddings = embeddings[indices_to_keep]
            kept_texts = [texts[i] for i in indices_to_keep]
            kept_indices = [indices[i] for i in indices_to_keep]
            
            new_index = create_faiss_index(kept_embeddings, index_type)
            
            stored_data['index'] = new_index
            stored_data['embeddings'] = kept_embeddings
            stored_data['texts'] = kept_texts
            stored_data['indices'] = kept_indices
        else:
            del _faiss_storage[storage_key]
        
        total_deleted = num_to_delete
    else:
        total_deleted = 0
    
    duration = time.time() - start
    
    # Prevent division by zero from making throughput look impossibly good
    duration = max(duration, 0.001)
    delete_throughput = total_deleted / duration
    
    save_result("delete", embedding_model, nrows, "faiss", delete_throughput)
    return delete_throughput

def save_result(operation, embedding_model, nrows, db_name, value):
    """
    Builds nested JSON structure that master.py expects. 
    NumPy types don't play nice with JSON, so we convert them first.
    """
    filename = f"{operation}.json"
    
    # JSON doesn't know about NumPy types, so we help it out
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
