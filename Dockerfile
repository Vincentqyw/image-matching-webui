FROM python:3.12-slim

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git-lfs ffmpeg libsm6 libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
