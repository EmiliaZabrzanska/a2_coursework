#!/bin/bash
# Build and run the medical_imaging Docker container

set -e  # exit on any error

IMAGE_NAME="medical-imaging"

echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

echo ""
echo "Running tests in container..."
docker run --rm $IMAGE_NAME

echo ""
echo "Done. To open an interactive Python shell, run:"
echo "  docker run -it --rm $IMAGE_NAME python"