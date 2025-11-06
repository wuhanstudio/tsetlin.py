import os
import struct
import utime

from machine import Pin
from detector import EdgeDetector

noise_level=50
state_threshold=15

led = Pin(33, Pin.OUT)

if 'main.bin' in os.listdir():
    file_size = os.stat('main.bin')[6]
    data_len  = int(file_size / 4)
    
    input_file = open('main.bin', 'rb')
    i = 0

    input_file.seek(i * 4)
    data = struct.unpack('f', input_file.read(4))
    detector = EdgeDetector(i, data[0], state_threshold=state_threshold, noise_level=noise_level)

    while True:
        i = i + 1
        input_file.seek(i * 4)
        data = struct.unpack('f', input_file.read(4))
        detector.update(i, data[0])
        print(data[0])
        if(detector.ongoing_change ):
            led.off()
        else:
            led.on()
        if i == data_len:
            i = 0
        utime.sleep_ms(100)
else:
    print("Please Upload data")
