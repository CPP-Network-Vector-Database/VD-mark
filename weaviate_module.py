import os
import time
import json
import ast
import pandas as pd
from sentence_transformers import SentenceTransformer
import weaviate
import random

# Helper to load a subset of the dataset
def load_dataset(dataset, nrows):
    df = pd.read_csv(dataset)
    if nrows is not None and nrows < len(df):
        df = df.iloc[:nrows]
    return df

# Helper to get or create schema
def ensure_schema_no_index(client):
    ip_flow_schema = {
        "class": "IPFlow",
        "vectorizer": "none",
        "vectorIndexConfig": {
            "skip": True  # Disable indexing during insertion
        },
        "properties": [
            {"name": "flow_index", "dataType": ["int"]},
            {"name": "metadata", "dataType": ["text"]},
        ]
    }
    existing_schema = client.schema.get()
    # If 'IPFlow' exists, delete it first
    for cls in existing_schema.get("classes", []):
        if cls["class"] == "IPFlow":
            client.schema.delete_class("IPFlow")
            break
    # Create the class
    client.schema.create_class(ip_flow_schema)

# Helper to enable indexing after insertion
def enable_indexing(client):
    # Weaviate doesn't allow updating skip=True to skip=False
    # So we need to delete and recreate the class with indexing enabled
    ip_flow_schema = {
        "class": "IPFlow",
        "vectorizer": "none",
        "vectorIndexConfig": {
            "skip": False  # Enable indexing
        },
        "properties": [
            {"name": "flow_index", "dataType": ["int"]},
            {"name": "metadata", "dataType": ["text"]},
        ]
    }
    
    # Delete the existing class
    client.schema.delete_class("IPFlow")
    # Recreate with indexing enabled
    client.schema.create_class(ip_flow_schema)

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

# Optimized CREATE operation: insert first, then build index
def create_vectors(embedding_model, dataset, nrows):
    client = weaviate.Client("http://localhost:8081")
    
    # Step 1: Create schema with indexing disabled
    ensure_schema_no_index(client)
    df = load_dataset(dataset, nrows)
    
    print(f"Inserting {len(df)} vectors without indexing...")
    start_time = time.time()
    
    # Step 2: Insert all data using batch for better performance
    batch_size = 100
    inserted_data = []  # Store data for re-insertion
    
    with client.batch as batch:
        batch.configure(batch_size=batch_size)
        
        for _, row in df.iterrows():
            vector = parse_vector(row["vector"])
            if vector is None:
                print(f"Skipping row {row['index']} due to vector parsing error")
                continue
                
            data_object = {
                "flow_index": int(row["index"]),
                "metadata": row["metadata"]
            }
            
            # Store for re-insertion with indexing
            inserted_data.append((data_object, vector))
            batch.add_data_object(data_object, "IPFlow", vector=vector)
    
    print("Insertion complete. Enabling indexing and re-inserting data...")
    
    # Step 3: Enable indexing (this deletes and recreates the class)
    enable_indexing(client)
    
    # Step 4: Re-insert data with indexing enabled
    with client.batch as batch:
        batch.configure(batch_size=batch_size)
        for data_object, vector in inserted_data:
            batch.add_data_object(data_object, "IPFlow", vector=vector)
    
    # Step 5: Wait for indexing to complete
    poll_start = time.time()
    while True:
        try:
            count = client.query.aggregate("IPFlow").with_meta_count().do()["data"]["Aggregate"]["IPFlow"][0]["meta"]["count"]
            if count >= len(inserted_data):
                break
        except Exception:
            pass
        time.sleep(1)
    
    total_time = time.time() - start_time
    throughput = len(inserted_data) / total_time
    
    print(f"Completed in {total_time:.2f} seconds")
    save_result("create", embedding_model, nrows, "weaviate", throughput)
    return throughput

# QUERY operation: run a semantic query and return average latency
def query_vectors(embedding_model, dataset, nrows, query_text=None, k=5):
    client = weaviate.Client("http://localhost:8081")

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
    result = (
        client.query
        .get("IPFlow", ["flow_index", "metadata"])
        .with_near_vector({"vector": vector})
        .with_additional(["distance"])
        .with_limit(k)
        .do()
    )
    latency = time.time() - start
    
    # Process results
    results = []
    if result.get("data", {}).get("Get", {}).get("IPFlow"):
        for item in result["data"]["Get"]["IPFlow"]:
            similarity = 1 - item["_additional"]["distance"]
            results.append({
                "flow_index": item.get("flow_index"),
                "metadata": item.get("metadata"),
                "similarity": similarity
            })
    
    # Calculate metrics
    query_throughput = 1 / latency if latency > 0 else 0
    avg_similarity = sum(r["similarity"] for r in results) / len(results) if results else 0
    
    # Save metrics
    save_result("query_throughput", embedding_model, nrows, "weaviate", query_throughput)
    save_result("query_similarity", embedding_model, nrows, "weaviate", avg_similarity)
    
    return query_throughput, avg_similarity, results

def delete_vectors(embedding_model, dataset, nrows):
    client = weaviate.Client("http://localhost:8081")
    start = time.time()
    total_deleted = 0
    
    try:
        # Using cursor-based pagination to collect IDs more efficiently
        all_items = []
        cursor = None
        batch_size = 1000
        
        print(f"Collecting object IDs using cursor pagination...")
        
        while True:
            try:
                query = (client.query
                        .get("IPFlow", ["_additional { id }"])
                        .with_limit(batch_size))
                
                # Add cursor if we have one
                if cursor:
                    query = query.with_after(cursor)
                
                result = query.do()
                
                if not result or "data" not in result:
                    break
                    
                items = result.get("data", {}).get("Get", {}).get("IPFlow", [])
                
                if not items:
                    break
                
                all_items.extend(items)
                print(f"Collected {len(items)} IDs. Total: {len(all_items)}")
                
                # Get cursor for next page
                if len(items) < batch_size:
                    break
                    
                # Update cursor to the last item's ID
                cursor = items[-1]["_additional"]["id"]
                
                # # Optional limit to prevent excessive memory usage
                # if len(all_items) >= 50000:
                #     break
                    
            except Exception as e:
                print(f"Error in cursor pagination: {e}")
                break
        
        # random sampling and deletion logic
        if len(all_items) == 0:
            print("No items found to delete")
            return 0
            
        items_to_delete = random.sample(all_items, min(nrows, len(all_items)))
        print(f"Randomly selected {len(items_to_delete)} items for deletion")
        
        # Delete in batches
        delete_batch_size = 1000
        
        for i in range(0, len(items_to_delete), delete_batch_size):
            batch = items_to_delete[i:i + delete_batch_size]
            
            for obj in batch:
                try:
                    obj_id = obj.get("_additional", {}).get("id")
                    if obj_id:
                        client.data_object.delete(obj_id, class_name="IPFlow")
                        total_deleted += 1
                except Exception as e:
                    print(f"Error deleting object: {e}")
                    continue
            
            print(f"Deleted batch {i//delete_batch_size + 1}. Total deleted: {total_deleted}")
            time.sleep(0.1)  # Small delay between batches
    
    except Exception as e:
        print(f"Error in cursor-based delete operation: {e}")
    
    duration = time.time() - start
    delete_throughput = total_deleted / duration if duration > 0 else 0
    
    print(f"Delete operation completed. Deleted {total_deleted} objects in {duration:.2f} seconds")
    
    save_result("delete", embedding_model, nrows, "weaviate", delete_throughput)
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