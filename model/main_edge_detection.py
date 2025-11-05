import nilmtk
from nilmtk.timeframe import TimeFrame

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np


def find_steady_states(dataframe, min_n_samples=2, state_threshold=15, noise_level=70):
    """Finds steady states given a DataFrame of power.

    Parameters
    ----------
    dataframe: pd.DataFrame with DateTimeIndex
    min_n_samples(int): number of samples to consider constituting a
        steady state.
    stateThreshold: maximum difference between highest and lowest
        value in steady state.
    noise_level: the level used to define significant
        appliances, transitions below this level will be ignored.
        See Hart 1985. p27.

    Returns
    -------
    steady_states, transitions
    """
    # Tells whether we have both real and reactive power or only real power
    estimated_steady_power = 0
    last_steady_power = 0
    previous_measurement = 0

    # These flags store state of power
    instantaneous_change = False    # power changing this second
    ongoing_change = False          # power change in progress over multiple seconds

    index_transitions_start = []          # Indices to use in returned Dataframe
    index_transitions_end = []          # Indices to use in returned Dataframe

    index_steady_states = []
    transitions = []                # holds information on transitions
    steady_states = []              # steadyStates to store in returned Dataframe
    N = 0                           # N stores the number of samples in state

    tran_start_time = []
    tran_end_time = dataframe.iloc[0].name

    for index, row in dataframe.iterrows():
        row = row.to_frame().iloc[0]

        # Step 1: Initialization
        if index == dataframe.index[0]:
            previous_measurement = row.iloc[0]
            last_steady_power = row.iloc[0]
            continue

        # Step 2: this does the threshold test and then we sum the boolean
        this_measurement = row.iloc[0]
        state_change = np.fabs(this_measurement - previous_measurement)

        if np.sum(state_change > state_threshold):
            instantaneous_change = True
            tran_start_time.append(dataframe.index[dataframe.index.get_loc(row.index[0]) - 1])
        else:
            instantaneous_change = False

        # Step 3: Identify if transition is just starting, if so, process it
        if instantaneous_change and (not ongoing_change):

            # Calculate transition size
            last_transition = estimated_steady_power - last_steady_power

            # Sum Boolean array to verify if transition is above noise level
            if np.fabs(last_transition) > noise_level:
                # 3A, C: if so add the index of the transition start and the power information

                # Avoid outputting first transition from zero
                index_transitions_end.append(tran_end_time)
                index_transitions_start.append(tran_start_time[0])
                tran_start_time = []
                transitions.append(last_transition)

                # I think we want this, though not specifically in Hart's algo notes
                # We don't want to append a steady state if it's less than min samples in length.
                # if N > min_n_samples:
                index_steady_states.append(tran_end_time)

                # last states steady power
                steady_states.append(estimated_steady_power)

            # 3B
            last_steady_power = estimated_steady_power
            # 3C
            # tran_end_time = row.index[0]  # start new steady state

        # Step 4: if a new steady state is starting, zero counter
        if instantaneous_change:
            N = 0

        # Hart step 5: update our estimate for steady state's energy
        estimated_steady_power = (N * estimated_steady_power + this_measurement) / (
            N + 1
        )

        # Step 6: increment counter
        N += 1

        # Step 7
        if ongoing_change and (not instantaneous_change):
            tran_end_time = dataframe.index[dataframe.index.get_loc(row.index[0]) - 1]
        ongoing_change = instantaneous_change

        # Step 8
        previous_measurement = this_measurement

    # Appending last edge
    last_transition = estimated_steady_power - last_steady_power
    if np.fabs(last_transition) > noise_level:
        index_transitions_start.append(tran_start_time[0])
        index_transitions_end.append(tran_end_time)
        transitions.append(last_transition)
    
        index_steady_states.append(tran_end_time)
        steady_states.append(estimated_steady_power)

    if len(index_transitions_end) == 0:
        # No events
        return pd.DataFrame(), pd.DataFrame()
    else:
        transients = pd.DataFrame({
            "active transition": transitions,
            "start time": index_transitions_start,
            "end time": index_transitions_end
        })
        steady_states = pd.DataFrame(
            data=steady_states, index=index_steady_states, columns=["active average"]
        )
        return steady_states, transients


TRAINING_DATA_DURATION = 1
delta_time = pd.to_timedelta(TRAINING_DATA_DURATION, unit="h")

# Load REDD dataset
redd = nilmtk.DataSet("model/redd.h5")
building_1 = redd.buildings[1].elec
main_meter = building_1.mains()[1]

# Get timeframe for training data
start_time = main_meter.get_timeframe().start
end_time = start_time + delta_time

kw = {
    "sections": [TimeFrame(start=start_time, end=end_time)],
    "sample_period": 3,
    "resample": True,
}

main_df = main_meter.power_series_all_data(**kw)
main_df = main_df.to_frame().fillna(0)

fridge = building_1["fridge"]
fridge_df = fridge.power_series_all_data(**kw)
fridge_df = fridge_df.to_frame().fillna(0)

# Global variables
noise_level = 70
state_threshold = 15

# Main: Find steady states
steady_states, transients = find_steady_states(
    main_df, noise_level=noise_level, state_threshold=state_threshold
)

# Plot steady states with main
ax = main_df.plot()
steady_states.plot(style="o", ax=ax)
for _, tran in transients.iterrows():
    plt.axvline(x=tran["start time"], color='r', linestyle='--', label='Important Date')
    # plt.axvline(x=tran["end time"], color='g', linestyle='--', label='Important Date')

plt.title("Steady states plot with main power signature")
plt.legend(["Mains", "Steady states"])
plt.ylabel("Power (W)")
plt.xlabel("Time")
plt.show()

# Fridge: Find steady states
steady_states_fridge, transients_fridge = find_steady_states(
    fridge_df, noise_level=noise_level, state_threshold=state_threshold
)

# Plot steady states with fridge
ax = fridge_df.plot()
steady_states_fridge.plot(style="o", ax=ax)
for _, tran in transients_fridge.iterrows():
    plt.axvline(x=tran["start time"], color='r', linestyle='--', label='Important Date')
    # plt.axvline(x=tran["end time"], color='g', linestyle='--', label='Important Date')

plt.title("Steady states plot with fridge power signature")
plt.legend(["Fridge", "Steady states"])
plt.ylabel("Power (W)")
plt.xlabel("Time")
plt.show()
