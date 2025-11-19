
import glob
import pandas as pd
from tqdm import tqdm
from loguru import logger

from detector import EdgeDetector
from nilmtk.timeframe import TimeFrame

NUM_BUILDINGS = [1, 2, 3, 4, 5, 6]
TRAINING_DATA_DURATION = 48  # hours
delta_time = pd.to_timedelta(TRAINING_DATA_DURATION, unit="h")

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

for i in range(len(NUM_BUILDINGS)):
    # Process each building
    building_id = i + 1
    logger.info(f"Processing Building {building_id}")

    # Pattern for files starting with 'redd_house_1' and ending with .csv
    file_pattern = f"redd_house{building_id}_*.csv"

    # Get list of matching files
    csv_files = glob.glob("redd/" + file_pattern)

    print("Files found:", csv_files)

    # Read and concatenate
    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

    appliance_names = df.columns.tolist()
    if 'fridge' not in appliance_names:
        logger.warning(f"Building {building_id} does not have a fridge. Skipping...")
        continue

    if 'microwave' not in appliance_names:
        logger.warning(f"Building {building_id} does not have a microwave. Skipping...")
        continue

    
    df = df.bfill()

    # Get main meter data
    main_df = df[["main"]]

    # Get fridge data
    fridge_df = df[["fridge"]]

    # # Get microwave data
    microwave_df = df[["microwave"]]

    # Get edge detection results
    logger.info(f"Performing edge detection for Building {building_id} main meter...")
    main_transient, main_steady = edge_detection(main_df, noise_level=80, state_threshold=15)

    logger.info(f"Performing edge detection for Building {building_id} fridge...")
    fridge_transient, fridge_steady = edge_detection(fridge_df, noise_level=80, state_threshold=15)

    logger.info(f"Performing edge detection for Building {building_id} microwave...")
    microwave_transient, microwave_steady = edge_detection(microwave_df, noise_level=80, state_threshold=15)

    main_transient.to_csv(f"building_{building_id}_main_transients.csv", index=False)
    # main_steady.to_csv(f"building_{building_id}_main_steady_states.csv", index=False)

    fridge_transient.to_csv(f"building_{building_id}_fridge_transients.csv", index=False)
    # fridge_steady.to_csv(f"building_{building_id}_fridge_steady_states.csv", index=False)

    microwave_transient.to_csv(f"building_{building_id}_microwave_transients.csv", index=False)
    # # microwave_steady.to_csv(f"building_{building_id}_microwave_steady_states.csv", index=False)
