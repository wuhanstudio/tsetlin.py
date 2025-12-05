import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from trace_pb2 import Traces, Trace

threshold = -1
epochs = 10
for epoch in range(epochs):
    traces = Traces()
    with open(f"traces_{threshold}_{epoch}.pb", "rb") as f:
            traces.ParseFromString(f.read())

    type_I_df = []
    type_II_df = []
    for i, trace in enumerate(tqdm(traces.trace)):
        if trace.type == Trace.Type.TYPE_I:
            type_I_df.append(trace.state)
        elif trace.type == Trace.Type.TYPE_II:
            type_II_df.append(trace.state)

    # Plot a histogram of the states
    plt.hist(type_I_df, bins=10, align='left', rwidth=0.8)
    plt.xlabel('State')
    plt.xlim(0, 50)
    plt.vlines(x=25, ymin=0, ymax=plt.ylim()[1], colors='r', linestyles='dashed', label='Midpoint')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Type I States - Epoch {epoch}')
    plt.show()

    plt.hist(type_II_df, bins=10, align='left', rwidth=0.8)
    plt.xlabel('State')
    plt.xlim(0, 50)
    plt.vlines(x=25, ymin=0, ymax=plt.ylim()[1], colors='r', linestyles='dashed', label='Midpoint')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Type II States - Epoch {epoch}')
    plt.show()
