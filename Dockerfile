# Use an official conda-based Python image as a parent image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL maintainer vincentqyw
ARG PYTHON_VERSION=3.10.4
# Set the working directory to /code
WORKDIR /code
# Copy the current directory contents into the container at /code
COPY . /code

RUN conda create -n imw python=${PYTHON_VERSION}
RUN echo "source activate imw" > ~/.bashrc
ENV PATH /opt/conda/envs/imw/bin:$PATH

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "imw", "/bin/bash", "-c"]
RUN pip install --upgrade pip

#RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r env-docker.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Export port
EXPOSE 7860