# pytlsd
Python transparent bindings for LSD (Line Segment Detector)

Bindings over the original C implementation of LSD, that allows to change the different thresholds involved and to provide custom image gradientes in order to compute the method with stronger features.

![](resources/example.jpg)

## Install
The current instructions were tested under Ubuntu 22.04:

```
sudo apt-get install build-essential cmake libopencv-dev libopencv-contrib-dev libarpack++2-dev libarpack2-dev libsuperlu-dev
git clone --recursive https://github.com/iago-suarez/pytlsd.git
cd pytlsd
pip3 install -r requirements.txt
pip3 install .
```

## Execution

```
python3 tests/test.py
```
