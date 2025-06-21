import os
import time
import json
import ast
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import random
import numpy as np

# --- Connection Details ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "IPFlow"
INDEX_TYPE = "IVF_FLAT"
METRIC_TYPE = "L2"

def load_dataset(dataset, nrows):
    df = pd.read_csv(dataset)
    return df.iloc[:nrows] if nrows and nrows < len(df) else df

def ensure_collection(embedding_model):
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    if embedding_model == "all-MiniLM-L6-v2":
        dim = 384
    elif embedding_model == "all-mpnet-base-v2":
        dim = 768
    elif embedding_model == "snowflake-arctic-embed-l-v2.0":
        dim = 1024
    else:
        raise ValueError(f"Unsupported embedding model: {embedding_model}")

    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="flow_index", dtype=DataType.INT64),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="IPFlow collection")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    index_params = {
        "metric_type": METRIC_TYPE,
        "index_type": INDEX_TYPE,
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    collection.load()
    return collection

def parse_vector(vector_str):
    try:
        return ast.literal_eval(vector_str.strip())
    except Exception as e:
        print(f"Vector parse error: {e}")
        return None

def create_vectors(embedding_model, dataset, nrows):
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = ensure_collection(embedding_model)
    df = load_dataset(dataset, nrows)

    start_time = time.time()
    entities = []

    for _, row in df.iterrows():
        vector = parse_vector(row["vector"])
        if vector is None:
            print(f"Skipping row {row['index']} due to vector parse error")
            continue
        vector = np.array(vector, dtype=np.float32)
        vector = vector / np.linalg.norm(vector)
        entities.append({
            "flow_index": int(row["index"]),
            "metadata": str(row["metadata"]),
            "vector": vector.tolist()
        })

    BATCH_SIZE = 1000
    for i in range(0, len(entities), BATCH_SIZE):
        batch = entities[i:i + BATCH_SIZE]
        if batch:
            collection.insert(batch)
    collection.flush()

    total_time = time.time() - start_time
    throughput = len(entities) / total_time if total_time > 0 else 0
    save_result("create", embedding_model, nrows, "milvus", throughput)
    connections.disconnect("default")
    return throughput

def query_vectors(embedding_model, dataset, nrows, query_text=None, k=5):
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    if not utility.has_collection(COLLECTION_NAME):
        print("Collection missing. Creating...")
        connections.disconnect("default")
        create_vectors(embedding_model, dataset, nrows)
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    collection = Collection(COLLECTION_NAME)
    collection.load()

    if collection.num_entities == 0:
        print("Collection empty. Creating...")
        connections.disconnect("default")
        create_vectors(embedding_model, dataset, nrows)
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(COLLECTION_NAME)
        collection.load()

    hf_model = "Snowflake/snowflake-arctic-embed-l-v2.0" if embedding_model == "snowflake-arctic-embed-l-v2.0" else embedding_model
    model = SentenceTransformer(hf_model)
    query_text = query_text or "Protocol with TCP"

    vector = model.encode(query_text)
    vector = vector / np.linalg.norm(vector)
    search_vector = vector.tolist()

    search_params = {"metric_type": METRIC_TYPE, "params": {"nprobe": 16}}
    k = min(k, collection.num_entities)

    if k <= 0:
        connections.disconnect("default")
        return 0, 0, []

    start = time.time()
    try:
        result = collection.search(
            data=[search_vector],
            anns_field="vector",
            param=search_params,
            limit=k,
            output_fields=["flow_index", "metadata"]
        )
        latency = time.time() - start
    except Exception as e:
        print(f"Search failed: {e}")
        connections.disconnect("default")
        return 0, 0, []

    results = []
    if result and len(result) > 0:
        for hit in result[0]:
            similarity = max(0, 1 - (hit.distance / 2))
            results.append({
                "flow_index": hit.entity.get("flow_index"),
                "metadata": hit.entity.get("metadata"),
                "similarity": similarity
            })

    query_throughput = 1 / latency if latency > 0 else 0
    avg_similarity = sum(r["similarity"] for r in results) / len(results) if results else 0

    save_result("query_throughput", embedding_model, nrows, "milvus", query_throughput)
    save_result("query_similarity", embedding_model, nrows, "milvus", avg_similarity)
    connections.disconnect("default")
    return query_throughput, avg_similarity, results

def delete_vectors(embedding_model, dataset, nrows):
    from pymilvus import connections, utility, Collection

    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    if not utility.has_collection(COLLECTION_NAME):
        print("Collection missing. Creating...")
        connections.disconnect("default")
        create_vectors(embedding_model, dataset, nrows)
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    collection = Collection(COLLECTION_NAME)
    collection.load()

    if collection.num_entities == 0:
        print("Collection empty. Creating...")
        connections.disconnect("default")
        create_vectors(embedding_model, dataset, nrows)
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(COLLECTION_NAME)
        collection.load()

    start = time.time()
    total_deleted = 0
    BATCH_SIZE = 5000  # Use smaller batches to avoid overloading system

    try:
        remaining_to_delete = min(nrows, collection.num_entities)

        while remaining_to_delete > 0:
            # Query a safe number of PKs
            limit = min(BATCH_SIZE, remaining_to_delete)
            res = collection.query(expr="", limit=limit, output_fields=["pk"])
            pks = [r["pk"] for r in res]

            if not pks:
                break  # No more PKs to delete

            collection.delete(expr=f"pk in {pks}")
            collection.flush()
            total_deleted += len(pks)
            remaining_to_delete -= len(pks)

    except Exception as e:
        print(f"Delete error: {e}")

    duration = time.time() - start
    delete_throughput = total_deleted / duration if duration > 0 else 0

    save_result("delete", embedding_model, nrows, "milvus", delete_throughput)
    connections.disconnect("default")
    return delete_throughput

def save_result(operation, embedding_model, nrows, db_name, value):
    filename = f"{operation}.json"
    value = float(value)

    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        data.setdefault(embedding_model, {}).setdefault(str(nrows), {})[db_name] = value

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Failed to save result to {filename}: {e}")