# Use an official conda-based Python image as a parent image
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
LABEL maintainer vincentqyw
ARG PYTHON_VERSION=3.10.10

# Set the working directory to /code
WORKDIR /code

# Install Git and Git LFS
RUN apt-get update && apt-get install -y git-lfs
RUN git lfs install

# Clone the Git repository
RUN git clone https://huggingface.co/spaces/Realcat/image-matching-webui /code

RUN conda create -n imw python=${PYTHON_VERSION}
RUN echo "source activate imw" > ~/.bashrc
ENV PATH /opt/conda/envs/imw/bin:$PATH

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "imw", "/bin/bash", "-c"]
RUN pip install --upgrade pip
RUN pip install -r env-docker.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Export port
EXPOSE 7860
