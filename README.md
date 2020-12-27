rm -r dist build app.spec
python3 -O -m PyInstaller --onefile --strip --debug bootloader --debug imports --upx-dir /home/pi/upx/ app.py
cp -r model dist/
/usr/local/bin/gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc -mfloat-abi=hard  --sysroot=$HOME/pluto-0.32.sysroot -std=c++11 -g -o pluto_stream program.cpp -lpthread -liio -lm -Wall -Wextra
iio_readdev -u ip:192.168.2.1 -b 256 -s 1024 cf-ad9361-lpc