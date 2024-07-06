# download submodules
git submodule update --init --recursive

# configuration envs
pip install -r env-docker.txt
apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
