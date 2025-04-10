# Use a base image with Python 3.10
FROM python:3.10-slim

# Set work directory inside the container
WORKDIR /app

# System dependencies for building packages like numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the code
COPY . .

# Run Makefile targets to set up the virtual environment and install dependencies
RUN make venv && \
    . .venv/bin/activate && \
    make install
# Set the environment to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Install Jupyter and essential packages
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipykernel \
    notebook \
    pandas \
    numpy \
    matplotlib

# Register the kernel with Jupyter
RUN python -m ipykernel install --user --name=python3

# Set up environment variables for proper PyCharm integration
ENV PYTHONUNBUFFERED=1

# Expose the Jupyter port
EXPOSE 8888

# Default command (interactive shell)
CMD ["/bin/bash"]