import os
import time
import json
import ast
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PointIdsList
import random

# Helper to load a subset of the dataset
def load_dataset(dataset, nrows):
    df = pd.read_csv(dataset)
    if nrows is not None and nrows < len(df):
        df = df.iloc[:nrows]
    return df

def ensure_collection(client, collection_name, vector_size):
    # Try to get the collection first
    try:
        client.get_collection(collection_name)
        # If it exists, delete it
        client.delete_collection(collection_name)
    except Exception:
        # Collection doesn't exist, we'll create it below
        pass
    
    # Create the collection (whether it existed before or not)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

# Helper to parse vector string to list of floats
def parse_vector(vector_str):
    """Parse the vector string from CSV to a list of floats"""
    try:
        # Remove any extra whitespace and parse as Python literal
        vector_list = ast.literal_eval(vector_str.strip())
        return vector_list
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing vector: {e}")
        return None

# CREATE operation: ingest vectors and return index construction time (to last object searchable)
def create_vectors(embedding_model, dataset, nrows, batch_size=500):
    client = QdrantClient("http://localhost:6333")
    df = load_dataset(dataset, nrows)
    
    # Find first valid vector to determine size (using generator for efficiency)
    vector_size = next(
        (len(vector) for vector in (parse_vector(row["vector"]) for _, row in df.iterrows()) 
         if vector is not None),
        None
    )
    
    if vector_size is None:
        raise ValueError("Could not determine vector size from dataset")
    
    collection_name = "IPFlow"
    ensure_collection(client, collection_name, vector_size)
    
    start_time = time.time()
    points = []
    
    # Pre-allocate list size if possible (assuming most vectors are valid)
    points = [None] * len(df)
    valid_count = 0
    
    for idx, row in df.iterrows():
        vector = parse_vector(row["vector"])
        if vector is None:
            print(f"Skipping row {row['index']} due to vector parsing error")
            continue
            
        points[valid_count] = PointStruct(
            id=int(row["index"]),
            vector=vector,
            payload={
                "flow_index": int(row["index"]),
                "metadata": row["metadata"]
            }
        )
        valid_count += 1
        
        # Batch upsert when batch size is reached
        if valid_count % batch_size == 0:
            client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points[valid_count-batch_size:valid_count]
            )
    
    # Upsert any remaining points
    if valid_count % batch_size != 0:
        client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points[valid_count-(valid_count%batch_size):valid_count]
        )
    
    total_time = time.time() - start_time
    throughput = valid_count / total_time
    save_result("create", embedding_model, nrows, "qdrant", throughput)
    return throughput

# QUERY operation: run a semantic query and return average latency
def query_vectors(embedding_model, dataset, nrows, query_text=None, k=5):
    client = QdrantClient("http://localhost:6333")

    # Map model alias to Hugging Face model ID
    if embedding_model == "snowflake-arctic-embed-l-v2.0":
        hf_model_name = "Snowflake/snowflake-arctic-embed-l-v2.0"
    else:
        hf_model_name = embedding_model
        
    model = SentenceTransformer(hf_model_name)
    
    # If no query text provided, use a default
    if query_text is None:
        query_text = "Protocol with TCP"
    
    # Encode the query text
    vector = model.encode(query_text).tolist()
    
    start = time.time()
    result = client.search(
        collection_name="IPFlow",
        query_vector=vector,
        limit=k
    )
    latency = time.time() - start
    
    # Process results
    results = []
    for hit in result:
        results.append({
            "flow_index": hit.payload.get("flow_index"),
            "metadata": hit.payload.get("metadata"),
            "similarity": hit.score
        })
    
    # Calculate metrics
    query_throughput = 1 / latency if latency > 0 else 0
    avg_similarity = sum(r["similarity"] for r in results) / len(results) if results else 0
    
    # Save metrics
    save_result("query_throughput", embedding_model, nrows, "qdrant", query_throughput)
    save_result("query_similarity", embedding_model, nrows, "qdrant", avg_similarity)
    
    return query_throughput, avg_similarity, results

# DELETE operation: delete all objects and return time taken
def delete_vectors(embedding_model, dataset, nrows):
    client = QdrantClient("http://localhost:6333")
    collection_name = "IPFlow"
    start = time.time()
    total_deleted = 0
    
    # Get all points and randomly select nrows to delete
    all_points = client.scroll(
        collection_name=collection_name,
        limit=nrows,
        with_payload=False,
        with_vectors=False
    )
    point_ids = [point.id for point in all_points[0]]
    
    # Randomly select nrows items if we have more
    if len(point_ids) > nrows:
        point_ids = random.sample(point_ids, nrows)
    
    # Delete the selected points
    if point_ids:
        client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=point_ids),
            wait=True
        )
        total_deleted = len(point_ids)
    
    duration = time.time() - start
    # Calculate delete throughput (deletions per second)
    delete_throughput = total_deleted / duration if duration > 0 else 0
    
    save_result("delete", embedding_model, nrows, "qdrant", delete_throughput)
    return delete_throughput

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
    if str(nrows) not in data[embedding_model]:
        data[embedding_model][str(nrows)] = {}
    data[embedding_model][str(nrows)][db_name] = value
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)