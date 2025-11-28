import serial
ser = serial.Serial('/dev/ttyS9', 115200, timeout=0, write_timeout=0.1)
ser.write(b'\x55\x55\x01\x55\x55')
print("write done")
