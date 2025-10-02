import time
import nilmtk

from flask import Flask

SAMPLE_PERIOD = 3  # seconds

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

redd = nilmtk.DataSet("redd.h5")

start = time.time()
building_1 = redd.buildings[1]

mains = building_1.elec.mains()
main_meter_0 = mains.meters[0]

main_meter_0_df = list(main_meter_0.load(sample_period=SAMPLE_PERIOD))[0]

@app.route("/building_1", methods=['GET'])
def building_1():
    global start
    end = time.time()

    main_meter_index = int((end - start) / SAMPLE_PERIOD)
    if main_meter_index >= len(main_meter_0_df):
        start = time.time()
        main_meter_index = 0
    
    return str(main_meter_0_df.iloc[main_meter_index][0])
