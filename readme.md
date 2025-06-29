# VD-mark - Vector Database Benchmarking Tool

`VD-mark` is a specialized benchmarking tool built to evaluate and compare the performance of various vector databases specifically for **network-related data** such as **IP flows**, **logs**, and **packets**.

It enables quick experimentation across multiple vector stores, allowing teams building AI-driven security, monitoring, or traffic analysis tools to identify the most optimal backend for their use case.


## Features

-  Benchmark performance of vector operations (insert, query, delete)
-  Focused on real-world network data like:
   - IP Flows
   - Network Logs
   - Packet Metadata
-  Supports multiple vector stores via modular plug-and-play architecture
-  Integrated observability with Prometheus + Grafana stack


## Modes of Operation

The VD-mark tool supports two primary modes for flexible usage depending on the need for automation or real-time inspection.

- Automated Mode 
- Interactive Mode (UI Mode)


##  Architecture

The architecture of the `VD-mark` Vector Benchmarking Tool follows a modular and containerized layout to ensure plug-and-play support for multiple vector databases and observability components.

>  **Diagram**: For visual representation, refer to the image below:

![Architecture Diagram](./assets/VD-mark.svg)


## Supported Vector Databases

Each database is handled via a dedicated module for ease of integration and extension:

- `chroma_module.py` - [Chroma DB](https://www.trychroma.com/)
- `faiss_module.py` - [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- `milvus_module.py` - [Milvus](https://milvus.io/)
- `pgvector_module.py` - [pgvector (PostgreSQL extension)](https://github.com/pgvector/pgvector)
- `qdrant_module.py` - [Qdrant](https://qdrant.tech/)
- `weaviate_module.py` - [Weaviate](https://weaviate.io/)

You can add your own by implementing the same function interfaces:  
`create_vectors`, `query_vectors`, and `delete_vectors`.


## Observability Stack

Integrated observability using:

- **Prometheus** – Metrics collection
- **Grafana** – Metrics dashboarding
- **Node Exporter** – System-level resource usage
- **cAdvisor** – Container-level monitoring

All components are Dockerized for quick deployment.


## Benchmark Scenarios

- Insertion throughput & latency
- Query performance with varying top-K and distances
- Delete operations
- Embedding storage formats and dimension handling
- Scalability with batch operations


## Additional Application Features & Capabilities

- **Interactive Streamlit UI:**
  - Configure benchmarks, select databases, embedding models, and dataset sizes.
  - Visualize results with interactive charts, tables, and downloadable CSVs.
  - Explore trends, compare models, and analyze performance metrics in real time.
- **Automated Benchmarking:**
  - Use `automatic_runner.py` for batch, reproducible experiments across datasets, models, and databases.
  - Results are saved and organized for later analysis.
- **Custom Dataset & Model Support:**
  - Easily add your own CSV datasets and select from multiple embedding models.
- **Advanced Visualization:**
  - Performance, trends, and explorer tabs for deep analysis.
  - Downloadable results for offline review.
- **Modular Database Integration:**
  - Add new vector DBs by implementing the required interface in a new module.

## Component Overview

- **UI Layer:**
  - `app.py`: Streamlit dashboard for configuration, execution, and visualization.
- **Benchmark Engine:**
  - `master.py`: Core logic for running benchmarks and managing results.
  - `automatic_runner.py`: Automates large-scale, repeatable experiments.
- **Database Modules:**
  - Individual modules for each supported vector DB (e.g., `faiss_module.py`, `milvus_module.py`, etc.).
- **Deployment:**
  - `database_deployment/` and `observability_deployment/`: Docker Compose files and scripts for launching databases and monitoring stack.
- **Observability:**
  - Prometheus, Grafana, Node Exporter, cAdvisor, and Postgres Exporter for full-stack monitoring.
- **Configuration:**
  - YAML and shell scripts for easy setup, database, and monitoring configuration.

## Deployment & Usage

### 1. Launch Vector Databases & Observability Stack
- Navigate to `database_deployment/` and run:
  ```bash
  sudo docker-compose up -d
  ```
- For observability (Prometheus, Grafana, etc.), use:
  ```bash
  cd ../observability_deployment
  sudo docker-compose up -d
  ```

### 2. Start/Stop All Services
- Use `start.sh` to bring up all database containers and wait for health checks.
- Use `clear_persistence.sh` to remove persistent volumes (use with caution).

### 3. Run the Streamlit UI
- From the `VD-mark` directory:
  ```bash
  streamlit run app.py
  ```
- Access the UI at `http://localhost:8501` by default.

### 4. Automated Benchmarking
- Run large-scale, automated experiments:
  ```bash
  python automatic_runner.py
  ```
- Results are saved and organized by dataset and operation.

## Extending the Tool

- **Add a New Vector Database:**
  - Create a new module (e.g., `mydb_module.py`) implementing `create_vectors`, `query_vectors`, and `delete_vectors`.
  - Register the module in `app.py` and `master.py`.
- **Add New Benchmark Scenarios:**
  - Extend the logic in `master.py` or `automatic_runner.py` to add new operations or metrics.

## Example Workflows

- **Interactive Benchmarking:**
  1. Launch all services (databases, observability).
  2. Start the Streamlit UI.
  3. Select databases, embedding models, dataset, and operation.
  4. Run benchmarks and explore results in the UI.

- **Automated Batch Benchmarking:**
  1. Prepare datasets in the required format.
  2. Run `automatic_runner.py` to execute all combinations.
  3. Review results in the generated folders or load them in the UI.

## Observability & Monitoring

- **Prometheus** collects metrics from all vector DB containers and system exporters.
- **Grafana** provides dashboards for real-time and historical performance analysis.
- **Node Exporter** and **cAdvisor** monitor system and container resource usage.
- **Postgres Exporter** tracks PostgreSQL metrics (for pgvector).
- Configuration files for Prometheus and exporters are in `observability_deployment/`.
- Access Grafana at `http://localhost:3000` (default credentials: `admin`/`admin`).

## Configuration Files

- **Database Configurations:**
  - `database_deployment/configuration/` contains YAML and conf files for each DB (e.g., Qdrant, Redis).
- **Observability Configurations:**
  - `observability_deployment/prometheus.yml`, `postgres_exporter.yml`, etc.
- **Docker Compose:**
  - `database_deployment/docker-compose.yml` and `observability_deployment/docker-compose.yml` define all services and their dependencies.

---
## Dataset

The dataset used for benchmarking in this setup is the IP Flows dataset, which can be found on [Kaggle](https://www.kaggle.com/datasets/ajayks23/packet-flow-datasets).

For further details, see the inline comments in each script and the deployment and observability READMEs in their respective folders.


