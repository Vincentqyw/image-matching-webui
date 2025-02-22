docker run -it -d \
  -p 7860:7860 \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8265:8265 \
  --name imc-test \
  vincentqin/image-matching-webui:latest

# docker run -it --entrypoint /code/start.sh image-matching-webui:latest
# docker run -it --entrypoint /bin/bash image-matching-webui:latest
