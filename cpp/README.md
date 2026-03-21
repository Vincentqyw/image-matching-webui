# C++ Code

This directory contains C++ code for the Image Matching WebUI project.

## Requirements

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    libcurl4-openssl-dev \
    libjsoncpp-dev \
    libb64-dev \
    libopencv-dev \
    libboost-all-dev \
    cmake
```

### macOS
```bash
brew install cmake opencv boost jsoncpp curl
```

## Build and Run

### Option 1: Using build script
```bash
cd cpp/test
mkdir -p build && cd build
cmake ..
make -j$(nproc)
./client
```

### Option 2: Using the provided script
```bash
cd cpp/test
bash build_and_run.sh
```

## Notes

- The client expects an API server running at `http://127.0.0.1:8001/v1/extract`
- Test images are located at `../../imcui/datasets/`
