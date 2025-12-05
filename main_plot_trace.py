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

    # Draw histogram
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(type_I_df, bins=10, edgecolor='black')

    # Total number of samples
    total = counts.sum()

    # Add percentages above bars
    for count, bin_left, bin_right in zip(counts, bins[:-1], bins[1:]):
        if count == 0:
            continue  # skip empty bins
        x = (bin_left + bin_right) / 2  # center of bar
        y = count
        percent = (count / total) * 100
        ax.text(x, y, f"{percent:.2f}%", ha='center', va='bottom', fontsize=8)

    plt.xlabel("State")
    plt.xlim(0, 50)
    plt.vlines(25, ymin=0, ymax=max(counts), colors='r', linestyles='dashed', label='middle point')
    plt.ylabel("Frequency")
    plt.title("Histogram of Type I States with Percentages")
    plt.show()

    # Draw histogram
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(type_II_df, bins=10, edgecolor='black')
    # Total number of samples
    total = counts.sum()

    # Add percentages above bars
    for count, bin_left, bin_right in zip(counts, bins[:-1], bins[1:]):
        if count == 0:
            continue  # skip empty bins
        x = (bin_left + bin_right) / 2  # center of bar
        y = count
        percent = (count / total) * 100
        ax.text(x, y, f"{percent:.2f}%", ha='center', va='bottom', fontsize=8)
    
    plt.xlabel("State")
    plt.xlim(0, 50)
    plt.vlines(25, ymin=0, ymax=max(counts), colors='r', linestyles='dashed', label='middle point')
    plt.ylabel("Frequency")
    plt.title("Histogram of Type II States with Percentages")
    plt.show()
