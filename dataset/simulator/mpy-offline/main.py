import os
import struct
import utime

if 'main.bin' in os.listdir():
    file_size = os.stat('main.bin')[6]
    data_len  = int(file_size / 4)
    
    input_file = open('main.bin', 'rb')
    i = 0
    while True:
        input_file.seek(i * 4)
        data = struct.unpack('f', input_file.read(4))
        print(data[0])
        i = i + 1
        if i == data_len:
            i = 0
        utime.sleep_ms(3000)
else:
    print("Please Upload data")
