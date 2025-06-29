import os
import time
import json
import ast
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import random

_faiss_storage = {}


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


app = Flask(__name__)


@app.route("/create_vectors", methods=["POST"])
def create_vectors_api():
    try:
        # Extract file and form data
        file = request.files.get("file")
        embedding_model = request.form.get("embedding_model")
        index_type = request.form.get("index_type", "IVFFLAT")
        nrows = int(request.form.get("nrows"))
        dataset = request.form.get("dataset")

        if not file or not embedding_model:
            return jsonify({"error": "Missing required fields"}), 400

        # Load the first N rows only from the uploaded CSV
        df = pd.read_csv(file, nrows=nrows)

        start_time = time.time()

        # Process and normalize embeddings
        embeddings, data_texts, data_indices = process_csv_data(df)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Create FAISS index
        index = create_faiss_index(embeddings, index_type)
        total_time = time.time() - start_time
        throughput = len(embeddings) / total_time if total_time > 0 else 0

        # Store in memory
        storage_key = get_storage_key(embedding_model, dataset, nrows, index_type)

        if storage_key in _faiss_storage:
            del _faiss_storage[storage_key]
            
        _faiss_storage[storage_key] = {
            "index": index,
            "embeddings": embeddings,
            "texts": data_texts,
            "indices": data_indices,
            "index_type": index_type
        }

        return jsonify({"throughput": throughput})

    except Exception as e:
        return jsonify({"error": f"Exception: {str(e)}"}), 500


@app.route("/query_vectors", methods=["POST"])
def query_vectors_api():
    try:
        data = request.get_json()

        embedding_model = data.get("embedding_model") 
        nrows = int(data.get("nrows"))
        dataset = data.get("dataset")
        query_text = data.get("query_text", "Protocol with TCP")
        k = data.get("k", 5)
        index_type = data.get("index_type", "IVFFLAT")

        if not all([embedding_model, dataset, nrows]):
            return jsonify({"error": "Missing required fields"}), 400

        storage_key = get_storage_key(embedding_model, dataset, nrows, index_type)

        if storage_key not in _faiss_storage:
            return jsonify({"error": "Index not found. Run /create_vectors"}), 404

        stored_data = _faiss_storage[storage_key]
        index = stored_data['index']
        texts = stored_data['texts']
        indices = stored_data['indices']

        # Resolve model name
        hf_model_name = (
            "Snowflake/snowflake-arctic-embed-l-v2.0"
            if embedding_model == "snowflake-arctic-embed-l-v2.0"
            else embedding_model
        )
        model = SentenceTransformer(hf_model_name)

        query_vector = model.encode(query_text)
        query_embedding = np.array([query_vector], dtype=np.float32)
        query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True)

        if hasattr(index, 'nprobe'):
            index.nprobe = min(10, getattr(index, 'nlist', 10))

        k = min(k, index.ntotal, len(texts))

        start = time.time()
        distances, faiss_indices = index.search(query_embedding, k)
        latency = time.time() - start

        results = []
        for dist, idx in zip(distances[0], faiss_indices[0]):
            if 0 <= idx < len(texts):
                similarity = max(0, 1 - (dist / 2))
                results.append({
                    "flow_index": int(indices[idx]),
                    "metadata": texts[idx],
                    "similarity": float(similarity)
                })

        query_throughput = 1 / latency if latency > 0 else 0
        avg_similarity = sum(r["similarity"] for r in results) / len(results) if results else 0

        

        return jsonify({
            "throughput": float(query_throughput),
            "avg_similarity": float(avg_similarity),
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/delete_vectors", methods=["POST"])
def delete_vectors_api():
    data = request.get_json()

    embedding_model = data["embedding_model"]
    nrows = data["nrows"]
    index_type = data.get("index_type", "IVFFLAT")
    dataset = data["dataset"]

    storage_key = get_storage_key(embedding_model, dataset, nrows, index_type)

    if storage_key not in _faiss_storage:
        return jsonify({"error": "Index not found. Run /create_vectors"}), 404

    stored_data = _faiss_storage[storage_key]

    # Check if HNSW index (doesn't support deletion)
    if index_type == 'HNSW':
        # For HNSW, we simulate deletion by clearing the storage
        start = time.time()
        total_vectors = stored_data['index'].ntotal
        num_to_delete = min(nrows, total_vectors)

        # Clear the storage to simulate deletion
        del _faiss_storage[storage_key]
        duration = time.time() - start
        delete_throughput = num_to_delete / duration if duration > 0 else 0

        return jsonify({"throughput": delete_throughput})

    index = stored_data['index']
    embeddings = stored_data['embeddings']
    texts = stored_data['texts']
    indices = stored_data['indices']

    start = time.time()

    # Get total available vectors
    total_vectors = len(embeddings)
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
            del _faiss_storage[storage_key]

        total_deleted = num_to_delete
    else:
        total_deleted = 0

    # Add minimum duration to prevent extremely high values
    duration = max(time.time() - start, 0.001)
    # Calculate delete throughput (deletions per second)
    delete_throughput = total_deleted / duration

    return jsonify({"throughput": delete_throughput})

if __name__ == '__main__':
    host = '0.0.0.0'
    port = 7000
    app.run(host=host, port=port, debug=True)