# download submodules
git submodule update --init --recursive

# configuration envs
pip install -r env-docker.txt
sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 -y