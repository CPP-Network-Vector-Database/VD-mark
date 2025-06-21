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


