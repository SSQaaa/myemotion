import serial
import time



ser_32_0 = serial.Serial('/dev/ttyS0', 115200, timeout=0.1) 
ser_51_4 = serial.Serial('/dev/ttyS4', 115200, timeout=0.1)
ser_mipi_3 = serial.Serial('/dev/ttyS3', 115200, timeout=0.1)

def send_32_0(data_byte):
    packet_0 = bytes([data_byte, data_byte, data_byte, data_byte, data_byte])
    ser_32_0.write(packet_0)
    print(f"[S0] Sent: {packet_0.hex()}")

def send_51_4(data_byte):
    packet_4 = str(data_byte) 
    ser_51_4.write(packet_4.encode("utf-8"))
    print(f"[S4] Sent: {packet_4}")

def send_51_4_str(data_str):

    ser_51_4.write(data_str.encode("utf-8"))
    print(f"[S4] Sent: {data_str}")

def send_mipi_3(data_byte):
    packet_3 = f"page {str(data_byte)}\r\n"
    ser_mipi_3.write(packet_3.encode("utf-8"))
    print(f"[S3] Sent: {packet_3.strip()}")

if __name__ == '__main__':
    # send_51_4(2)
    # send_mipi_3(0)



    # send_32_0(0x05)
    # send_51_4_str('b')
    # send_51_4_str('d')
    # send_32_0(0x07)
    # send_32_0(0x01)
    # send_51_4_str('c')
    # send_51_4(8)
    send_51_4_str('a')

    time.sleep(0.1) 
