import os
import time
import json
import ast
import pandas as pd
import numpy as np
import requests
import io

API_URL = "http://localhost:7000"

# Helper to load a subset of the dataset
def load_dataset(dataset, nrows):
    df = pd.read_csv(dataset)
    if nrows is not None and nrows < len(df):
        df = df.iloc[:nrows]
    return df

# CREATE operation: ingest vectors and return index construction time (to last object searchable)
def create_vectors(embedding_model, dataset, nrows, index_type='IVFFLAT'):
    try:
        # Load only nrows from large file
        df = pd.read_csv(dataset, nrows=nrows)

        # Write those rows to an in-memory CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Prepare multipart/form-data
        files = {
            "file": ("subset.csv", csv_buffer, "text/csv")
        }
        data = {
            "embedding_model": embedding_model,
            "nrows": str(nrows),
            "index_type": index_type,
            "dataset": dataset
        }

        # Send POST request
        response = requests.post(f"{API_URL}/create_vectors", data=data, files=files)
        response.raise_for_status()

        throughput = response.json().get("throughput", 0.0)
        save_result("create", embedding_model, nrows, "faiss", throughput)

        return throughput

    except Exception as e:
        print(f"[Client] Create failed: {e}")
        return 0.0

# QUERY operation: run a semantic query and return average latency
def query_vectors(embedding_model, dataset, nrows, query_text=None, k=5, index_type='IVFFLAT'):
    payload = {
        "embedding_model": embedding_model,
        "nrows": nrows,
        "query_text": query_text or "Protocol with TCP",
        "dataset": dataset,
        "k": k,
        "index_type": index_type
    }

    try:
        response = requests.post(f"{API_URL}/query_vectors", json=payload)

        # Handle 404: Retry by first creating the vectors
        if response.status_code == 404:
            error_msg = response.json().get("error", "")
            if "Index not found" in error_msg:
                print("Index not found. Calling create_vectors and retrying...")
                create_vectors(embedding_model, dataset, nrows, index_type)

                # Retry query after creation
                response = requests.post(f"{API_URL}/query_vectors", json=payload)
                response.raise_for_status()
            else:
                response.raise_for_status()
        else:
            response.raise_for_status()

        data = response.json()

        query_throughput = data.get("throughput", 0.0)
        avg_similarity = data.get("avg_similarity", 0.0)
        results = data.get("results", [])

        save_result("query_throughput", embedding_model, nrows, "faiss", query_throughput)
        save_result("query_similarity", embedding_model, nrows, "faiss", avg_similarity)

        return query_throughput, avg_similarity, results

    except requests.exceptions.RequestException as e:
        print(f"Query API call failed: {e}")
        return 0.0, 0.0, []

# DELETE operation: delete all objects and return time taken
def delete_vectors(embedding_model, dataset, nrows, index_type='IVFFLAT'):
    payload = {
        "embedding_model": embedding_model,
        "nrows": nrows,
        "dataset": dataset,
        "index_type": index_type
    }

    try:
        response = requests.post(f"{API_URL}/delete_vectors", json=payload)

        if response.status_code == 404:
            error_msg = response.json().get("error", "")
            if "Index not found" in error_msg:
                print("Index not found. Calling create_vectors before deletion...")
                create_vectors(embedding_model, dataset, nrows, index_type)

                # Retry deletion after creation
                response = requests.post(f"{API_URL}/delete_vectors", json=payload)
                response.raise_for_status()
            else:
                response.raise_for_status()
        else:
            response.raise_for_status()

        data = response.json()
        delete_throughput = data.get("throughput", 0.0)

        save_result("delete", embedding_model, nrows, "faiss", delete_throughput)
        return delete_throughput

    except requests.exceptions.RequestException as e:
        print(f"Delete API call failed: {e}")
        return 0.0


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