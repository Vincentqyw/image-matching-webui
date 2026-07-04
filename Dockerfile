FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime
LABEL maintainer vincentqyw

WORKDIR /code

# all together
RUN apt-get update && apt-get install -y --no-install-recommends \
    git-lfs ffmpeg libsm6 libxext6 && \
    git lfs install && \
    git config --global url."https://github.com/".insteadOf git@github.com: && \
    git clone --recursive https://github.com/Vincentqyw/image-matching-webui.git . && \
    pip install --no-cache-dir --upgrade --break-system-packages pip && \
    pip install --no-cache-dir --break-system-packages -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip cache purge

EXPOSE 7860 8000 8001 8265
