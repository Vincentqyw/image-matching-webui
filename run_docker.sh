docker run --gpus all --ipc=host -v $(pwd):/host -p 7860:7860  -it image-matching-webui python /code/app.py
