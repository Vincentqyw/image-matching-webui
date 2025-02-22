# Use official conda base image
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
LABEL maintainer vincentqyw

# Set working directory
WORKDIR /code

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Initialize Git LFS
RUN git lfs install

# Clone repository
RUN git clone --recursive https://github.com/Vincentqyw/image-matching-webui.git /code

# Configure conda environment
RUN conda create -n imw python=3.10.10 && \
    echo "source activate imw" >> ~/.bashrc
ENV PATH /opt/conda/envs/imw/bin:$PATH

# Set conda environment as default execution context
SHELL ["conda", "run", "-n", "imw", "/bin/bash", "-c"]

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy startup script
COPY docker/start.sh /code/start.sh
RUN chmod +x /code/start.sh

# Expose service ports
EXPOSE 7860 8000 8001 8265

# Launch services
CMD ["/code/start.sh"]
