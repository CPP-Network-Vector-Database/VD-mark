# Database Deployment

## Usage Instructions

### Docker Compose: Up & Down

#### Start Docker Containers

To spin up the Docker containers defined in the `docker-compose.yml` file, run the following command:

```bash
sudo docker-compose up -d
```

This will start the containers in detached mode (i.e., in the background).

#### Stop Docker Containers

To bring the containers down and stop all running services, use the following command:

```bash
sudo docker-compose down
```

This command will stop the containers and remove the containers, networks, and volumes defined in your `docker-compose.yml` file.

---

### Clear Persistence Script

The `clear_persistence.sh` script is designed to remove Docker volumes attached to stopped containers. This can be useful if you want to free up storage space or clear persistent data stored in volumes.

#### Usage:

To run the script and clear persistent volumes, execute the following command:

```bash
bash clear_persistence.sh
```

This command will:

- Check for all stopped containers.
- Identify volumes that are attached to those stopped containers.
- Remove the volumes associated with the stopped containers to free up storage.

> **Note:** Use this script with caution, as it will permanently delete the volumes and their data.
