from master import run_benchmark
import os
import shutil
import logging

logging.basicConfig(
    filename='master.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define dataset types, embedding models, rows, and databases
DATASET_TYPES = ['ipflows', 'iplogs']
EMBEDDING_MODELS = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'snowflake-arctic-embed-l-v2.0']
ROWS = [2000, 4000, 6000, 8000, 10000, 20000, 50000, 100000, 200000]
DATABASES = ['pgvector', 'weaviate', 'milvus','qdrant', 'chroma', 'faiss']

FILES_TO_MOVE = [
    'create.json',
    'delete.json',
    'query.json',
    'query_similarity.json',
    'query_throughput.json'
]

for dataset_type in DATASET_TYPES:
    for embedding_model in EMBEDDING_MODELS:
        dataset_name = f'{dataset_type}_200000_{embedding_model}.csv'
        
        for row_count in ROWS:
            # CREATE operation
            create_results = run_benchmark(
                selected_databases=DATABASES,
                selected_embedding_models=[embedding_model],
                selected_dataset=dataset_name,
                selected_sizes=[row_count],
                operation='create',
                faiss_index_type='IVFFLAT'
            )
            
            # QUERY operation
            query_results = run_benchmark(
                selected_databases=DATABASES,
                selected_embedding_models=[embedding_model],
                selected_dataset=dataset_name,
                selected_sizes=[row_count],
                operation='query',
                faiss_index_type='IVFFLAT',
                query_text='TCP Protocol',
                k_results=5
            )
            
            # DELETE operation
            delete_results = run_benchmark(
                selected_databases=DATABASES,
                selected_embedding_models=[embedding_model],
                selected_dataset=dataset_name,
                selected_sizes=[row_count],
                operation='delete',
                faiss_index_type='IVFFLAT'
            )
            
            msg = f"Completed for dataset: {dataset_type}, model: {embedding_model}, rows: {row_count}"
            print(msg)
            logger.info(msg)
    
    # Move files after finishing dataset_type
    target_folder = dataset_type
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    for file_name in FILES_TO_MOVE:
        if os.path.exists(file_name):
            shutil.move(file_name, os.path.join(target_folder, file_name))
            msg = f"Moved {file_name} to folder {target_folder}"
            print(msg)
            logger.info(msg)
        else:
            warn_msg = f"Warning: {file_name} not found, skipping move."
            print(warn_msg)
            logger.warning(warn_msg)