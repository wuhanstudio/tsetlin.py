import nilmtk
from nilmtk.timeframe import TimeFrame

import pandas as pd
from tqdm import tqdm
from loguru import logger

from detector import EdgeDetector

def edge_detection(dataframe, noise_level=50, state_threshold=15):
    detector = None
    with tqdm(total=dataframe.shape[0]) as pbar:
        for index, row in dataframe.iterrows():
            row = row.to_frame().iloc[0]
            current_time = row.index[0]
            current_measurement = row.iloc[0].item()

            # Initialize detector on first iteration
            if index == dataframe.index[0]:
                detector = EdgeDetector(current_time, current_measurement, state_threshold=state_threshold, noise_level=noise_level)
                continue

            output = detector.update(current_time, current_measurement)
            # if output.get('transition', False):
            #     logger.info(f"Duration: {len(output['transition_data'])} samples")
            #     logger.info(f"Transition: {output['transition_power_change']}")
            #     logger.info(f"Transition: {output['transition_data']}")
            #     logger.info("---")

            pbar.update(1)

    # Prepare DataFrames for steady states and transients
    steady_states = pd.DataFrame()
    transients = pd.DataFrame()

    assert len(detector.transitions) == len(detector.tran_data_list)

    # Create DataFrames if we have detected any transitions
    if len(detector.index_transitions_end) > 0:
        transients = pd.DataFrame({
            "transition": detector.transitions,
            "duration": [len(tran) for tran in detector.tran_data_list],
            "start": detector.index_transitions_start,
            "end": detector.index_transitions_end,
            "sequence": detector.tran_data_list
        })
        steady_states = pd.DataFrame(
            data=detector.steady_states, index=detector.index_steady_states, columns=["active average"]
        )
    
    return transients, steady_states

# Load REDD dataset
redd = nilmtk.DataSet("redd.h5")

for i in range(len(redd.buildings)):
    # Process each building
    building_id = i + 1
    logger.info(f"Processing Building {building_id}")

    building = redd.buildings[building_id].elec

    appliance_names = [ app.type['type'] for app in building.appliances]
    if 'fridge' not in appliance_names:
        logger.warning(f"Building {building_id} does not have a fridge. Skipping...")
        continue

    main_meter = building.mains()[1]

    start_time = main_meter.get_timeframe().start
    end_time = main_meter.get_timeframe().end

    kw = {
        "sections": [TimeFrame(start=start_time, end=end_time)],
        "sample_period": 3,
        "resample": True,
    }

    # Get main meter data
    main_df = main_meter.power_series_all_data(**kw)
    main_df = main_df.to_frame().fillna(0)

    # Get fridge data
    fridge = building["fridge"]
    fridge_df = fridge.power_series_all_data(**kw)
    fridge_df = fridge_df.to_frame().fillna(0)

    # Get edge detection results
    main_transient, main_steady = edge_detection(main_df, noise_level=80, state_threshold=15)
    fridge_transient, fridge_steady = edge_detection(fridge_df, noise_level=80, state_threshold=15)

    main_transient.to_csv(f"building_{building_id}_main_transients.csv", index=False)
    # main_steady.to_csv(f"building_{building_id}_main_steady_states.csv", index=False)

    fridge_transient.to_csv(f"building_{building_id}_fridge_transients.csv", index=False)
    # fridge_steady.to_csv(f"building_{building_id}_fridge_steady_states.csv", index=False)

redd.store.close()
