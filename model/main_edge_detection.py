import nilmtk
from nilmtk.timeframe import TimeFrame

from loguru import logger

import pandas as pd
import matplotlib.pyplot as plt

from detector import EdgeDetector

TRAINING_DATA_DURATION = 48  # hours
delta_time = pd.to_timedelta(TRAINING_DATA_DURATION, unit="h")

def plot_edge_detection(dataframe, noise_level=50, state_threshold=15):
    detector = None
    for index, row in dataframe.iterrows():
        row = row.to_frame().iloc[0]
        current_time = row.index[0]
        current_measurement = row.iloc[0].item()

        # Initialize detector on first iteration
        if index == dataframe.index[0]:
            detector = EdgeDetector(current_time, current_measurement, state_threshold=state_threshold, noise_level=noise_level)
            continue

        output = detector.update(current_time, current_measurement)
        if output.get('transition', False):
            logger.info(f"Duration: {len(output['transition_data'])} samples")
            logger.info(f"Transition: {output['transition_power_change']}")
            logger.info(f"Transition: {output['transition_data']}")
            logger.info("---")

    # Prepare DataFrames for steady states and transients
    steady_states = pd.DataFrame()
    transients = pd.DataFrame()

    assert len(detector.transitions) == len(detector.tran_data_list)

    # Create DataFrames if we have detected any transitions
    if len(detector.index_transitions_end) > 0:
        transients = pd.DataFrame({
            "active transition": detector.transitions,
            "start time": detector.index_transitions_start,
            "end time": detector.index_transitions_end
        })
        steady_states = pd.DataFrame(
            data=detector.steady_states, index=detector.index_steady_states, columns=["active average"]
        )

    # Plot steady states with main
    ax = dataframe.plot()
    if not steady_states.empty:
        steady_states.plot(style="o", ax=ax)
        for _, tran in transients.iterrows():
            plt.axvline(x=tran["start time"], color='r', linestyle='--', label='Start Time')
            # plt.axvline(x=tran["end time"], color='g', linestyle='--', label='End Time')

    plt.legend(["Measurement", "Steady states"])
    plt.ylabel("Power (W)")
    plt.xlabel("Time")
    plt.show()

# Load REDD dataset
redd = nilmtk.DataSet("redd.h5")
building_1 = redd.buildings[1].elec
main_meter = building_1.mains()[1]

# Get timeframe for training data
start_time = main_meter.get_timeframe().start
# start_time = pd.Timestamp('2011-04-18 18:30:54-0400', tz='US/Eastern')

end_time = start_time + delta_time
# end_time = main_meter.get_timeframe().end

kw = {
    "sections": [TimeFrame(start=start_time, end=end_time)],
    "sample_period": 3,
    "resample": True,
}

# Get main meter data
main_df = main_meter.power_series_all_data(**kw)
main_df = main_df.to_frame().fillna(0)

# Get fridge data
fridge = building_1["fridge"]

fridge_df = fridge.power_series_all_data(**kw)
fridge_df = fridge_df.to_frame().fillna(0)

# Get microwave data
microwave = building_1["microwave"]
microwave_df = microwave.power_series_all_data(**kw)
microwave_df = microwave_df.to_frame().fillna(0)

# Plot edge detection results
plot_edge_detection(main_df, noise_level=80, state_threshold=15)
plot_edge_detection(fridge_df, noise_level=80, state_threshold=15)
plot_edge_detection(microwave_df, noise_level=80, state_threshold=15)

redd.store.close()
