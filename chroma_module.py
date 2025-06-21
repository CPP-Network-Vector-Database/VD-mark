import os
import time
import json
import ast
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import numpy as np
import random

# ChromaDB specific constants
CHROMA_HOST = "localhost"
CHROMA_HTTP_PORT = 8000
COLLECTION_NAME = "chroma_"  # Consistent collection name

# Shared ChromaDB client - reused in all functions
shared_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_HTTP_PORT)

# Helper to load a subset of the dataset
def load_dataset(dataset_path, nrows):
    df = pd.read_csv(dataset_path)
    if nrows is not None and nrows < len(df):
        df = df.iloc[:nrows]
    return df

# Save result to JSON file in the nested structure expected by master.py
def save_result(operation, embedding_model, nrows, db_name, value):
    filename = f"{operation}.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    if embedding_model not in data:
        data[embedding_model] = {}
    nrows_str = str(nrows)
    if nrows_str not in data[embedding_model]:
        data[embedding_model][nrows_str] = {}
    data[embedding_model][nrows_str][db_name] = value
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

# Helper to parse vector string to list of floats
def parse_vector(vector_str):
    """Parse the vector string from CSV to a list of floats"""
    try:
        vector_list = ast.literal_eval(vector_str.strip())
        return vector_list
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing vector: {e}")
        return None

# CREATE operation: ingest vectors and return throughput
def create_vectors(embedding_model, dataset, nrows):
    collection_name = COLLECTION_NAME + embedding_model + "_" + dataset
    collection = shared_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    df = load_dataset(dataset, nrows)

    # Map model alias to Hugging Face model ID
    if embedding_model == "snowflake-arctic-embed-l-v2.0":
        hf_model_name = "Snowflake/snowflake-arctic-embed-l-v2.0"
    else:
        hf_model_name = embedding_model    

    model = SentenceTransformer(hf_model_name)

    all_ids = []
    all_embeddings = []
    all_metadatas = []

    start_time = time.time()

    for _, row in df.iterrows():
        all_embeddings.append(parse_vector(row["vector"]))
        all_metadatas.append({"metadata": row["metadata"]})
        all_ids.append(str(row["index"]))

    MAX_BATCH_SIZE = 500
    for i in range(0, len(all_ids), MAX_BATCH_SIZE):
        collection.add(
            ids=all_ids[i:i+MAX_BATCH_SIZE],
            embeddings=all_embeddings[i:i+MAX_BATCH_SIZE],
            metadatas=all_metadatas[i:i+MAX_BATCH_SIZE]
        )

    total_time = time.time() - start_time
    throughput = len(df) / total_time if total_time > 0 else 0
    save_result("create", embedding_model, nrows, "chroma", throughput)
    return throughput

# QUERY operation: run a semantic query and return throughput and average similarity
def query_vectors(embedding_model, dataset, nrows, query_text=None, k=5):
    try:
        collection_name = COLLECTION_NAME + embedding_model + "_" + dataset
        collection = shared_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    except Exception:
        save_result("query_throughput", embedding_model, nrows, "chroma", 0.0)
        save_result("query_similarity", embedding_model, nrows, "chroma", 0.0)
        return 0.0, 0.0, [] 

    if collection.count() == 0:
        save_result("query_throughput", embedding_model, nrows, "chroma", 0.0)
        save_result("query_similarity", embedding_model, nrows, "chroma", 0.0)
        return 0.0, 0.0, []

    if embedding_model == "snowflake-arctic-embed-l-v2.0":
        hf_model_name = "Snowflake/snowflake-arctic-embed-l-v2.0"
    else:
        hf_model_name = embedding_model

    model = SentenceTransformer(hf_model_name)

    if query_text is None:
        query_text = "Protocol with TCP"

    query_embedding = model.encode(query_text).tolist()

    start_time = time.time()

    query_response = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=['distances', 'metadatas']
    )

    latency = time.time() - start_time

    response_ids = query_response['ids'][0]
    response_distances = query_response['distances'][0]
    response_metadatas = query_response['metadatas'][0]

    processed_results = []
    for i in range(len(response_ids)):
        processed_results.append({
            "flow_index": int(response_ids[i]),
            "metadata": response_metadatas[i].get("metadata", {}),
            "similarity": 1 - response_distances[i]
        })

    avg_similarity = sum(item["similarity"] for item in processed_results) / len(processed_results) if processed_results else 0.0
    query_throughput = 1 / latency if latency > 0 else 0

    save_result("query_throughput", embedding_model, nrows, "chroma", query_throughput)
    save_result("query_similarity", embedding_model, nrows, "chroma", avg_similarity)

    return query_throughput, avg_similarity, processed_results

# DELETE operation: delete specified number of objects and return throughput
def delete_vectors(embedding_model, dataset, nrows):
    try:
        collection_name = COLLECTION_NAME + embedding_model + "_" + dataset
        collection = shared_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    except Exception:
        save_result("delete", embedding_model, nrows, "chroma", 0.0)
        return 0.0

    current_count = collection.count()
    if nrows > current_count:
        save_result("delete", embedding_model, nrows, "chroma", 0.0)
        return 0.0

    retrieved_data = collection.get(limit=current_count, include=[])
    all_ids_in_collection = retrieved_data['ids']
    ids_to_delete = random.sample(all_ids_in_collection, nrows)

    start_time = time.time()

    MAX_BATCH_SIZE = 500
    for i in range(0, len(ids_to_delete), MAX_BATCH_SIZE):
        collection.delete(ids=ids_to_delete[i:i + MAX_BATCH_SIZE])

    duration = time.time() - start_time
    delete_throughput = nrows / duration if duration > 0 else 0

    save_result("delete", embedding_model , nrows, "chroma", delete_throughput)
    return delete_throughput
