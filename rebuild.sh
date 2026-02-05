#!/bin/bash

# --- CONFIGURATION ---
IMAGE_NAME="logistics_price_etanol:latest"
CONTAINER_NAME="streamlit" # Renamed to be more descriptive
# Port mapping: HOST_PORT:CONTAINER_PORT
# Adjusted to 8013 to match your entrypoint.sh script
PORT_MAPPING="8013:8013"
# The WORKDIR from your Dockerfile. Adjust if it's different.
SOURCE_CODE_DIR="/app"


# --- COLORS ---
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

echo -e "${YELLOW}--- STARTING DEVELOPMENT ENVIRONMENT REBUILD ---${NC}"

# Step 1: Stop and remove the old container by name
# Usage of "|| true" is to prevent errors if the container doesn't exist
echo "=> Stopping and removing the old container ('${CONTAINER_NAME}')..."
docker stop ${CONTAINER_NAME} || true
docker rm ${CONTAINER_NAME} || true

# Step 2: Aggressive general cleanup of the Docker system
# Note: This is very aggressive and might be slow. You can comment this out for faster restarts.
echo "=> Removing all stopped containers, unused images, and volumes..."
docker system prune -a --volumes -f

echo -e "\\n${GREEN}--- CLEANUP COMPLETE ---${NC}"

echo -e "\\n${YELLOW}--- STARTING NEW DEVELOPMENT ENVIRONMENT ---${NC}"

# Step 3: Build the new image from the Dockerfile
# This is mainly for when you change dependencies (requirements.txt)
echo "=> Building the new image ('${IMAGE_NAME}')..."
docker build -t ${IMAGE_NAME} .

# Step 4: Start the new container in DEVELOPMENT mode
echo "=> Starting new container ('${CONTAINER_NAME}') with live-reload..."

docker run -d \
    --name ${CONTAINER_NAME} \
    -p ${PORT_MAPPING} \
    -e ENVIRONMENT="development" \
    -v "$(pwd)":${SOURCE_CODE_DIR} \
    ${IMAGE_NAME}

: '
# For debug container, example understand how is file directory tree
docker run -it --rm \
--name ${CONTAINER_NAME} \
-p ${PORT_MAPPING} \
-e PORT=8013 \
-v "$(pwd)":${SOURCE_CODE_DIR} \
${IMAGE_NAME} \
/bin/bash
'

: "
The '-it' = interactive mode.
The '--rm' = auto-delete the container when we're done.
The '-d' runs the container in 'detached mode (in the background)
The '--name' gives the container a fixed name, making it easy to remove the next time the script runs
The '-p' maps your computer's port (host) to the container's port
The '-e' sets the environment variable to trigger development mode in entrypoint.sh
The '-v' mounts your current directory into the container for live-reloading
The '/bin/bash' = the command to run (a shell instead of your app).
"
echo -e "\\n${GREEN}--- PROCESS COMPLETE! DEVELOPMENT ENVIRONMENT IS READY. ---${NC}"