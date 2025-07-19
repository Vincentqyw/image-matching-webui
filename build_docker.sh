# docker build -t image-matching-webui:latest . --no-cache
# docker tag image-matching-webui:latest vincentqin/image-matching-webui:latest
# docker push vincentqin/image-matching-webui:latest

#!/bin/bash
# build-and-push.sh

IMAGE_NAME="vincentqin/image-matching-webui"
VERSION="latest"

echo "🔨 Building image..."
docker build -t $IMAGE_NAME:$VERSION .

echo "📝 Tagging versions..."
# tagging to easily use
docker tag $IMAGE_NAME:$VERSION $IMAGE_NAME:$(date +%Y%m%d)
docker tag $IMAGE_NAME:$VERSION $IMAGE_NAME:v1.0

echo "🚀 Pushing to Docker Hub..."
docker push $IMAGE_NAME:$VERSION
docker push $IMAGE_NAME:$(date +%Y%m%d)
docker push $IMAGE_NAME:v1.0

echo "✅ Push completed!"
echo "📖 Usage:"
echo "  docker run -p 7860:7860 $IMAGE_NAME:$VERSION"
echo "🌐 Docker Hub: https://hub.docker.com/repository/docker/vincentqin/image-matching-webui"
