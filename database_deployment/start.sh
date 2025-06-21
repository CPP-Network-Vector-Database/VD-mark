#!/bin/bash

# Run docker-compose up -d
docker-compose up -d

# Wait for all containers in vector_dbs with health checks to become healthy
echo "Waiting for containers to become healthy..."

while true; do
  all_healthy=true

  for container in $(docker network inspect vector_dbs -f '{{range .Containers}}{{.Name}} {{end}}'); do
    health=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$container" 2>/dev/null)

    if [ "$health" == "healthy" ]; then
      echo "$container: true"
    elif [ "$health" == "starting" ] || [ "$health" == "unhealthy" ]; then
      echo "$container: false"
      all_healthy=false
    fi
    # If health == none, skip
  done

  if [ "$all_healthy" = true ]; then
    echo "Successfully done"
    break
  else
    echo "Waiting for containers to be healthy..."
    sleep 5
  fi
done