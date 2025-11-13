import os
import struct
import utime

import json
from machine import Pin
from detector import EdgeDetector

from tsetlin import Tsetlin
from tsetlin.utils.booleanize import booleanize_features

noise_level=80
state_threshold=15

MODEL_FILE_NAME = "tsetlin_model_redd.upb"

N_BIT = 8
X_mean = [10.644803469072505, 9.214285714285714]
X_std  = [1077.2628013914837, 4.46263979129692]

USE_TFT = False

if USE_TFT:
    from machine import SPI
    from ST7735 import TFT, TFTColor
    from sysfont import sysfont

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
    tft.text((2, 40), "Duration: ", TFT.RED, sysfont, 1, nowrap=True)
    tft.text((2, 60), "Transition", TFT.RED, sysfont, 1, nowrap=True)

if 'main.bin' in os.listdir():
    file_size = os.stat('main.bin')[6]
    data_len  = int(file_size / 4)
    
    main_file = open('main.bin', 'rb')
    fridge_file = open('fridge.bin', 'rb')

    i = 0
    main_file.seek(i * 4)
    main_data = struct.unpack('f', main_file.read(4))
    detector = EdgeDetector(i, main_data[0], state_threshold=state_threshold, noise_level=noise_level)

    t_model = Tsetlin.load_umodel(MODEL_FILE_NAME)

    while True:
        i = i + 1

        # Read data from file
        main_file.seek(i * 4)
        fridge_file.seek(i * 4)
        
        main_data = struct.unpack('f', main_file.read(4))
        fridge_data = struct.unpack('f', fridge_file.read(4))

        # Run edge detection
        output = detector.update(i, main_data[0])
        
        result = {}
        result["main"] = main_data[0]
        result["fridge"] = fridge_data[0]

        if output.get('transition', False):
            # Run Tsetlin Machine Clasifier        
            X = [output['transition_power_change'], len(output['transition_data'])]
            X_bool = booleanize_features([X], X_mean, X_std, num_bits=N_BIT)
            y_pred = t_model.predict(X_bool)
            
            result["transition"] = output['transition_power_change']
            result["duration"] = len(output['transition_data'])
            result["prediction"] = y_pred[0]

            # Display on TFT
            if USE_TFT:
                tft.text((2, 40), f"Duration: {len(output['transition_data'])} samples", TFT.RED, sysfont, 1, nowrap=True)
                tft.text((2, 60), f"Transition: {output['transition_power_change']}", TFT.RED, sysfont, 1, nowrap=True)
                tft.text((2, 80), "                                                                                ", TFT.RED, sysfont, 1, nowrap=True)
                tft.text((2, 80), f"{[ int(t) for t in output['transition_data'] ]}", TFT.RED, sysfont, 1, nowrap=True)
        
        print(json.dumps(result))

        if(detector.ongoing_change ):
            led.off()
        else:
            led.on()

        if i == data_len:
            i = 0

        utime.sleep_ms(100)
else:
    print("Please Upload data")
