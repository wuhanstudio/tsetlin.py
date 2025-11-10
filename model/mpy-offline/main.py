import os
import struct
import utime

from machine import Pin
from detector import EdgeDetector

from machine import SPI, Pin
from ST7735 import TFT, TFTColor

from sysfont import sysfont

noise_level=50
state_threshold=15

USE_TFT = True

led = Pin(33, Pin.OUT)

if USE_TFT:
    TFT_SCK  = 18
    TFT_MOSI = 23
    TFT_MISO = 19

    TFT_LED = 13
    TFT_RST = 16
    TFT_RS  = 17
    TFT_CS  = 26

    # Initialize TFT
    back_light = Pin(TFT_LED, Pin.OUT)
    back_light.on()

    spi = SPI(2, baudrate=20000000, polarity=0, phase=0, sck=Pin(TFT_SCK), mosi=Pin(TFT_MOSI), miso=Pin(TFT_MISO))
    tft=TFT(spi, TFT_RS, TFT_RST, TFT_CS)

    tft.initr()
    tft.rgb(True)

    tft.fill(TFT.BLACK)
    tft.text((15, 40), "Duration: ", TFT.RED, sysfont, 1, nowrap=True)
    tft.text((15, 60), "Transition", TFT.RED, sysfont, 1, nowrap=True)
    tft.text((15, 80), "Sequence: ", TFT.RED, sysfont, 1, nowrap=True)

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

        # Read data from file
        input_file.seek(i * 4)
        data = struct.unpack('f', input_file.read(4))

        print(data[0])

        # Run edge detection
        output = detector.update(i, data[0])
        
        # Display on TFT
        if output.get('transition', False) and USE_TFT:
            tft.text((15, 40), f"Duration: {len(output['transition_data'])} samples", TFT.RED, sysfont, 1, nowrap=True)
            tft.text((15, 60), f"Transition: {output['transition_power_change']}", TFT.RED, sysfont, 1, nowrap=True)
            tft.text((15, 80), f"Sequence: {output['transition_data']}", TFT.RED, sysfont, 1, nowrap=True)

        if(detector.ongoing_change ):
            led.off()
        else:
            led.on()

        if i == data_len:
            i = 0

        utime.sleep_ms(100)
else:
    print("Please Upload data")
