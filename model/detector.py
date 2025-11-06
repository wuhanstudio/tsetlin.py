import numpy as np
from collections import deque
from statistics import stdev

class EdgeDetector:
    def __init__(self, current_time, current_measurement, state_threshold=15, noise_level=70, min_n_samples=5):
        self.state_threshold = state_threshold
        self.noise_level = noise_level
        self.min_n_samples = min_n_samples

        # Tells whether we have both real and reactive power or only real power
        self.estimated_steady_power = 0
        self.instantaneous_change_dequeue = deque(maxlen=min_n_samples)
        self.last_steady_power = 0
        self.previous_measurement = 0

        # These flags store state of power
        self.instantaneous_change = False    # power changing this second
        self.ongoing_change = False          # power change in progress over multiple seconds

        self.index_transitions_start = []          # Indices to use in returned Dataframe
        self.index_transitions_end = []          # Indices to use in returned Dataframe

        self.index_steady_states = []
        self.transitions = []                # holds information on transitions
        self.steady_states = []              # steadyStates to store in returned Dataframe
        self.N = 0                           # N stores the number of samples in state

        self.tran_start_time = []
        self.tran_end_time = None

        self.previous_measurement = current_measurement
        self.last_steady_power = current_measurement

        self.previous_time = current_time

    def update(self, current_time, current_measurement):
        # Step 2: this does the threshold test and then we sum the boolean
        state_change = np.fabs(current_measurement - self.previous_measurement)

        if np.sum(state_change > self.state_threshold):
            if state_change > self.noise_level:
                self.tran_start_time.append(self.previous_time)
            self.instantaneous_change = True
        else:
            self.instantaneous_change = False

        # Identify end of ongoing change
        if self.ongoing_change and (not self.instantaneous_change):
            self.tran_end_time = self.previous_time

        # Step 3: Identify if transition is just starting, if so, process it
        # if instantaneous_change and (not ongoing_change):
        if len(self.instantaneous_change_dequeue) == self.min_n_samples and all(not x for x in self.instantaneous_change_dequeue):

            # Calculate transition size
            last_transition = self.estimated_steady_power - self.last_steady_power

            # Sum Boolean array to verify if transition is above noise level
            if np.fabs(last_transition) > self.noise_level:
                # 3A, C: if so add the index of the transition start and the power information

                # Avoid outputting first transition from zero
                self.index_transitions_end.append(self.tran_end_time)
                self.index_transitions_start.append(self.tran_start_time[0])
                self.tran_start_time = []
                self.transitions.append(last_transition)

                # I think we want this, though not specifically in Hart's algo notes
                # We don't want to append a steady state if it's less than min samples in length.
                # if N > min_n_samples:
                self.index_steady_states.append(self.tran_end_time)

                # last states steady power
                self.steady_states.append(self.estimated_steady_power)
            # else:
                # tran_start_time = []

            # 3B
            self.last_steady_power = self.estimated_steady_power

        # 3C
        # tran_end_time = row.index[0]  # start new steady state

        # Step 4: if a new steady state is starting, zero counter
        if self.instantaneous_change:
            self.N = 0

        # Hart step 5: update our estimate for steady state's energy
        self.estimated_steady_power = (self.N * self.estimated_steady_power + current_measurement) / (
            self.N + 1
        )
        self.instantaneous_change_dequeue.append(self.instantaneous_change)

        # Step 6: increment counter
        self.N += 1

        # Step 7
        self.ongoing_change = self.instantaneous_change

        # Step 8
        self.previous_measurement = current_measurement
        self.previous_time = current_time
