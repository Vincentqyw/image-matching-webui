# g++ main.cpp -I/usr/include/opencv4 -lcurl -ljsoncpp -lb64 -lopencv_core -lopencv_imgcodecs -o main
# sudo apt-get update
# sudo apt-get install libboost-all-dev -y
# sudo apt-get install libcurl4-openssl-dev libjsoncpp-dev libb64-dev libopencv-dev -y

cd build
cmake ..
make -j12

echo " ======== RUN DEMO ========"

./client

echo " ======== END DEMO ========"

cd ..
