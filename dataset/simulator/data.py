import nilmtk
import numpy as np
from tqdm import tqdm
from loguru import logger

SAMPLE_PERIOD = 6  # seconds

redd = nilmtk.DataSet("redd.h5")

building_1 = redd.buildings[1]

mains = building_1.elec.mains()
main_meter_0 = mains.meters[0]

main_meter_0_df = list(main_meter_0.load(sample_period=SAMPLE_PERIOD))[0]
main_meter_0_df.dropna(inplace=True)

# main_meter_0_df = main_meter_0_df.iloc[0:(len(main_meter_0_df) // 2)]

output_file = open('main.bin', 'wb')
main_meter_0_data = main_meter_0_df.values.flatten()
main_meter_0_data.tofile(output_file)
output_file.close()

logger.info(f"Wrote {len(main_meter_0_data)} * {main_meter_0_data.itemsize} samples to main.bin")

input_file = open('main.bin', 'rb')

# logger.info(f"Read {len(main_meter_0_data_from_file)} samples from main.bin")
# main_meter_0_data_from_file = np.fromfile(input_file, dtype=main_meter_0_data.dtype)

logger.info("Verifying data integrity...")

for i in tqdm(range(len(main_meter_0_data))):
    input_file.seek(i * main_meter_0_data.itemsize)
    sample = np.fromfile(input_file, dtype=main_meter_0_data.dtype, count=1)[0]

    assert np.isclose(sample, main_meter_0_data[i]), f"Data mismatch at index {i}, {sample} != {main_meter_0_data[i]}"

input_file.close()
redd.store.close()
