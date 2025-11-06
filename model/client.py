import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

import serial

ser = serial.Serial('COM12', 115200, timeout=1)

i = 0

# This function is called periodically from FuncAnimation
def animate(i, xs, ys):
    line = ser.readline().decode().strip()
    if line:
        try:
            main_meter = round(float(line), 2)
        except ValueError:
            print("Non-float data:", line)

    # Add x and y to lists
    xs.append(i)
    i = i + 1
    ys.append(main_meter)

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.set_ylim([0, 2000])
    ax.plot(xs, ys)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('Building 1 Main Meter')
    plt.ylabel('Power (W)')

if __name__ == '__main__':
    # Set up plot to call animate() function periodically
    ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=50)
    plt.show()
