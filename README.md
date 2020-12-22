rm -r dist build app.spec
python3 -O -m PyInstaller --onefile --strip --debug bootloader --debug imports --upx-dir /home/pi/upx/ app.py
cp -r model dist/