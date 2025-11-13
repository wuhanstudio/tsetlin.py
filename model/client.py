import time
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import json
import serial
from loguru import logger

SERIAL_PORT = "COM12"  # Update this to your serial port
ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)

time.sleep(10)  # wait for the serial connection to initialize

# Create figure for plotting
fig = plt.figure()
ax_main = fig.add_subplot(2, 1, 1)
ax_fridge = fig.add_subplot(2, 1, 2)

main_xs = []
main_ys = []

fridge_xs = []
fridge_ys = []

edge = []
fridge_edge = []

i = 0

# This function is called periodically from FuncAnimation
def animate(i, main_xs, main_ys, fridge_xs, fridge_ys, edge, fridge_edge):
    line = ser.readline().decode().strip()

    prediction = None
    duration = None
    transition = None

    if line:
        try:
            data = json.loads(line)
            logger.debug(f"{data}")

            main_meter = round(float(data.get("main", 0)), 2)
            fridge_meter = round(float(data.get("fridge", 0)), 2)

            if 'transition' in data:
                transition = data['transition']
                logger.debug(f"Power Change: {transition}")
            if 'duration' in data:
                duration = data['duration']
                logger.debug(f"Duration: {duration} samples")
            if 'prediction' in data:
                prediction = data['prediction']
                logger.debug(f"Prediction: {prediction}")

            # Add x and y to lists
            main_xs.append(i)
            main_ys.append(main_meter)

            fridge_xs.append(i)
            fridge_ys.append(fridge_meter)


            if transition is not None and duration is not None:
                edge.append(True)
            else:
                edge.append(False)

            if prediction is not None and prediction == 1:
                fridge_edge.append(True)
            else:
                fridge_edge.append(False)

            i = i + 1

            # Limit x and y lists to 20 items
            main_xs = main_xs[-20:]
            main_ys = main_ys[-20:]
            
            fridge_xs = fridge_xs[-20:]
            fridge_ys = fridge_ys[-20:]

            edge = edge[-20:]
            fridge_edge = fridge_edge[-20:]

            # Draw x and y lists
            ax_main.clear()
            ax_main.set_ylim([0, 2000])
            ax_main.plot(main_xs, main_ys)

            ax_fridge.clear()
            ax_fridge.set_ylim([0, 2000])
            ax_fridge.plot(fridge_xs, fridge_ys)

            ax_main.set_title('Building 1 Main Meter')
            ax_main.set_ylabel('Power (W)')

            ax_fridge.set_title('Building 1 Fridge Meter')
            ax_fridge.set_ylabel('Power (W)')

            if len(fridge_edge) > 0:
                for index, f in enumerate(fridge_edge):
                    if f:
                        ax_main.axvline(x = main_xs[index], color = 'r', linestyle='--', label='fridge')  # Vertical line at x=5

            for index, e in enumerate(edge):
                if e:
                    ax_main.plot(main_xs[index], main_ys[index], 'o', color='red', markersize=10)  # 'o' = circle marker

            # Format plot
            plt.xticks(rotation=45, ha='right')
            # plt.subplots_adjust(bottom=0.30)
            # plt.title('Building 1 Main Meter')
            # plt.ylabel('Power (W)')

        except (ValueError, json.JSONDecodeError):
            logger.warning(f"Non-float or invalid JSON data: {line}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

if __name__ == '__main__':
    # Set up plot to call animate() function periodically
    ani = animation.FuncAnimation(fig, animate, fargs=(main_xs, main_ys, fridge_xs, fridge_ys, edge, fridge_edge), interval=50)
    plt.show()
