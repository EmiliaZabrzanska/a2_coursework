# Stage 1: Build the package
FROM python:3.12-slim AS builder

WORKDIR /app

# Copy only what's needed for the build
COPY pyproject.toml README.md LICENSE ./
COPY medical_imaging/ medical_imaging/

# Install build tools and build the package
RUN pip install --upgrade pip setuptools wheel build \
    && python -m build

# Stage 2: Runtime image with only the installed package
FROM python:3.12-slim

WORKDIR /app

# Install system libraries required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 && rm -rf /var/lib/apt/lists/*

# Copy the built wheel from the builder stage
COPY --from=builder /app/dist/*.whl /tmp/

# Install the wheel (includes all dependencies from pyproject.toml)
RUN pip install --no-cache-dir /tmp/*.whl \
    && rm /tmp/*.whl

# Copy test files so we can run tests inside the container
COPY tests/ tests/

# Install pytest for running tests
RUN pip install --no-cache-dir pytest

# Default command: run the test suite
CMD ["pytest", "-s", "tests/"]