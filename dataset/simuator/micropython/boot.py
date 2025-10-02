import utime
import network
import urequests as requests

from machine import SPI,Pin
from ST7735 import TFT,TFTColor

from sysfont import sysfont

# Change to your settings
WIFI_SSID = 'nilm'
WIFI_PWD  = 'nonintrusiveloadmonitoring'

SERVER_URL = 'http://192.168.0.58:5000/building_1'

TFT_SCK  = 18
TFT_MOSI = 23
TFT_MISO = 19

TFT_LED = 13
TFT_RST = 16
TFT_RS  = 17
TFT_CS  = 26

# Connect to WIFI
sta_if = network.WLAN(network.STA_IF)
if not sta_if.active():
    sta_if.active(True)

if not sta_if.isconnected():
    sta_if.connect(WIFI_SSID, WIFI_PWD)
    for i in range(0, 5):
        if(not sta_if.isconnected()):
            utime.sleep_ms(1000)

print(sta_if.ifconfig())

# Initialize TFT
back_light = Pin(TFT_LED, Pin.OUT)
back_light.on()

spi = SPI(2, baudrate=20000000, polarity=0, phase=0, sck=Pin(TFT_SCK), mosi=Pin(TFT_MOSI), miso=Pin(TFT_MISO))
tft=TFT(spi, TFT_RS, TFT_RST, TFT_CS)

tft.initr()
tft.rgb(True)

tft.fill(TFT.BLACK)
tft.text((15, 40), "Building 1 - Main", TFT.RED, sysfont, 1, nowrap=True)

while True:

    res = requests.get(url=SEVER_URL)

    print(res.text)
    tft.text((35, 60), '            ', TFT.RED, sysfont, 1, nowrap=True)
    tft.text((35, 60), res.text, TFT.RED, sysfont, 1, nowrap=True)
    utime.sleep_ms(3000)

