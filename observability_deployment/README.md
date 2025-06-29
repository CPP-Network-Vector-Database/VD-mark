# Monitoring Stack with Prometheus, Grafana, and Exporters

This Stack provides a Docker Compose setup for a complete monitoring solution using Prometheus, Grafana, and common exporters for PostgreSQL, system metrics, and Docker containers.

## Services Included

### 1. Prometheus
Prometheus is a time-series database and monitoring system that collects metrics from configured targets.

- **Port:** `9090`
- **Config file:** `prometheus.yml`
- **Volumes:**
  - `./prometheus.yml:/etc/prometheus/prometheus.yml`
  - `./prometheus:/prometheus`

### 2. Grafana
Grafana is a visualization and analytics tool that connects to Prometheus and other data sources.

- **Port:** `3000`
- **Volumes:**
  - `./grafana:/var/lib/grafana`

### 3. PostgreSQL Exporter
The `postgres_exporter` exposes metrics from a PostgreSQL database for Prometheus to scrape.

- **Port:** `9187`
- **Environment Variable:**
  - `DATA_SOURCE_NAME`: Replace `{username}`, `{password}`, and `{host}` accordingly.
    ```
    postgresql://<username>:<password>@<host>:5432/vector?sslmode=disable
    ```
- **Volumes:**
  - `./postgres_exporter.yml:/etc/postgres_exporter/postgres_exporter.yml`

### 4. Node Exporter
The `node_exporter` exposes system-level metrics such as CPU, memory, disk, and network.

- **Port:** `9100`

### 5. cAdvisor
Google's cAdvisor provides container-level resource usage and performance metrics.

- **Port:** `8082`
- **Volumes:**
  - Host system directories mounted for real-time container monitoring.

## Getting Started

### Prerequisites
- Docker and Docker Compose installed on your machine.

### Run the Stack

```bash
docker-compose up -d