docker build -t image-matching-webui:latest . --no-cache
docker tag image-matching-webui:latest vincentqin/image-matching-webui:latest
docker push vincentqin/image-matching-webui:latest
