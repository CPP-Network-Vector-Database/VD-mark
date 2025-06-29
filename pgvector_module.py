import os
import time
import json
import random
import ast
import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer
import re
from typing import List, Union
import torch

# Helper to load a subset of the dataset
def load_dataset(dataset, nrows):
    df = pd.read_csv(dataset)
    if nrows is not None and nrows < len(df):
        df = df.iloc[:nrows]
    return df

# Connect to PostgreSQL
def connect_db():
    return psycopg2.connect(
        dbname="vector",
        user="username", # Change it 
        password="password", # Change it 
        host="localhost",
        port="5432"
    )

def ensure_schema(conn, embedding_model):
    embedding_dimensions = {
        'all-MiniLM-L6-v2': 384,
        'all-mpnet-base-v2': 768,
        'snowflake-arctic-embed-l-v2.0': 1024,
    }

    dim = embedding_dimensions.get(embedding_model)
    if dim is None:
        raise ValueError(f"Unknown embedding model dimension for: {embedding_model}")

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("DROP TABLE IF EXISTS ipflow;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS ipflow (
                id SERIAL PRIMARY KEY,
                flow_index INT,
                metadata TEXT,
                embedding vector({dim})
            );
        """)
    conn.commit()

def create_index(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE INDEX IF NOT EXISTS hnsw_idx 
            ON ipflow USING hnsw (embedding vector_cosine_ops) 
            WITH (ef_construction = 200, m = 16);
        """)
    conn.commit()

# Helper to parse vector string to list of floats
def parse_vector(vector_str):
    try:
        vector_list = ast.literal_eval(vector_str.strip())
        return vector_list
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing vector: {e}")
        return None

# CREATE operation: insert vectors and measure throughput
def create_vectors(embedding_model, dataset, nrows):
    conn = connect_db()
    ensure_schema(conn, embedding_model)

    df = load_dataset(dataset, nrows)
    start_time = time.time()
    with conn.cursor() as cur:
        for _, row in df.iterrows():
            metadata = row["metadata"]
            vector_str = row["vector"]
            vector = parse_vector(vector_str)
            if vector is None:
                print(f"Skipping row {row['index']} due to vector parsing error")
                continue
            cur.execute("""
                INSERT INTO ipflow (flow_index, metadata, embedding)
                VALUES (%s, %s, %s);
            """, (int(row["index"]), metadata, vector))
    conn.commit()

    duration = time.time() - start_time
    
    # Create index after bulk insert
    create_index(conn)
    
    throughput = len(df) / duration if duration > 0 else 0
    save_result("create", embedding_model, nrows, "pgvector", throughput)
    conn.close()
    return throughput

# QUERY operation: run a semantic search and return throughput and similarity
def query_vectors(embedding_model, dataset, nrows, query_text=None, k=5):
    conn = connect_db()

    # Map model alias to Hugging Face model ID
    if embedding_model == "snowflake-arctic-embed-l-v2.0":
        hf_model_name = "Snowflake/snowflake-arctic-embed-l-v2.0"  # or "all-MiniLM-L6-v2"
    else:
        hf_model_name = embedding_model

    model = SentenceTransformer(hf_model_name)

    if query_text is None:
        query_text = "Protocol with TCP"
    vector = model.encode(query_text).tolist()

    start = time.time()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT flow_index, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM ipflow
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (vector, vector, k))
        results = cur.fetchall()
    latency = time.time() - start

    output = []
    for row in results:
        output.append({
            "flow_index": row[0],
            "metadata": row[1],
            "similarity": row[2]
        })

    avg_similarity = sum(r["similarity"] for r in output) / len(output) if output else 0
    query_throughput = 1 / latency if latency > 0 else 0

    save_result("query_throughput", embedding_model, nrows, "pgvector", query_throughput)
    save_result("query_similarity", embedding_model, nrows, "pgvector", avg_similarity)
    conn.close()
    return query_throughput, avg_similarity, output

# DELETE operation: delete nrows random objects and return throughput
def delete_vectors(embedding_model, dataset, nrows):
    conn = connect_db()
    
    total_deleted = 0

    with conn.cursor() as cur:
        # Fetch up to nrows IDs
        cur.execute("SELECT id FROM ipflow LIMIT %s;", (nrows,))
        items = [row[0] for row in cur.fetchall()]

        # Randomly sample if more than needed
        if len(items) > nrows:
            items = random.sample(items, nrows)

        start = time.time()
        for obj_id in items:
            cur.execute("DELETE FROM ipflow WHERE id = %s;", (obj_id,))
            total_deleted += 1

    conn.commit()
    duration = time.time() - start
    delete_throughput = total_deleted / duration if duration > 0 else 0
    save_result("delete", embedding_model, nrows, "pgvector", delete_throughput)
    conn.close()
    return delete_throughput

# Save result to JSON file in nested structure
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