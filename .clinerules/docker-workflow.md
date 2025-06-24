## Brief overview
This file documents the standard workflow and commands for interacting with the Dockerized development environment for this project.

## Core Commands
- **Convention**: Always use `docker compose` (with a space) instead of the legacy `docker-compose` (with a hyphen).
- **Building the Image**: To build or rebuild the Docker image after changing `Dockerfile` or `requirements.txt`.
  - `docker compose -f docker/docker-compose.yml build`
- **Starting the Environment**: To start the development container in the background.
  - `docker compose -f docker/docker-compose.yml up -d`
- **Stopping the Environment**: To stop and remove the development container.
  - `docker compose -f docker/docker-compose.yml down`

## Running Scripts from Host
- The primary workflow is to execute scripts inside the container directly from the host's terminal, without entering the container's shell.
- **Example (running a training script):**
  - `docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_bc.py`

## GUI Forwarding (X11)
- To enable GUI applications from the container to display on the host, the following setup is required:
  - **Host Permission**: The host machine must grant permission to the container. This is done by running the following command once in the host's terminal:
    - `xhost +local:docker`
  - **Compose Configuration**: The `docker-compose.yml` file is configured with `network_mode: "host"` and the `DISPLAY` environment variable (e.g., `DISPLAY=:1`) to facilitate the connection.

## Accessing the Shell (for Debugging)
- While not the primary workflow, you can get an interactive bash shell for debugging purposes if needed.
  - `docker compose -f docker/docker-compose.yml exec dvrk-dev /bin/bash`
