FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /code

# all together
RUN apt-get update && apt-get install -y --no-install-recommends \
    git-lfs ffmpeg libsm6 libxext6 && \
    git lfs install && \
    git clone --recursive https://github.com/Vincentqyw/image-matching-webui.git . && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip cache purge

EXPOSE 7860 8000 8001 8265
