import re
import matplotlib.pyplot as plt

# --- Sample Log Data ---
# You can add more log lines as needed.
log_data = """
24-10-08 14:22:26.145 : Epoch: 2/100, Learning Rate: 0.00020 | Training, AVG_Loss 0.6904, AVG_ACC: 53.3500 | Validating, AVG_Loss 0.6766, AVG_ACC: 56.6429
24-10-08 14:25:27.397 : Epoch: 4/100, Learning Rate: 0.00020 | Training, AVG_Loss 0.6257, AVG_ACC: 61.4500 | Validating, AVG_Loss 0.5828, AVG_ACC: 69.4286
24-10-08 14:28:24.596 : Epoch: 6/100, Learning Rate: 0.00020 | Training, AVG_Loss 0.7163, AVG_ACC: 57.6500 | Validating, AVG_Loss 0.6379, AVG_ACC: 68.3571
24-10-08 14:31:22.972 : Epoch: 8/100, Learning Rate: 0.00020 | Training, AVG_Loss 0.6624, AVG_ACC: 58.0833 | Validating, AVG_Loss 0.6774, AVG_ACC: 56.2143
24-10-08 14:34:22.277 : Epoch: 10/100, Learning Rate: 0.00020 | Training, AVG_Loss 0.6251, AVG_ACC: 60.6833 | Validating, AVG_Loss 0.5909, AVG_ACC: 70.5000
24-10-08 14:37:24.716 : Epoch: 12/100, Learning Rate: 0.00020 | Training, AVG_Loss 0.5783, AVG_ACC: 68.0167 | Validating, AVG_Loss 0.5883, AVG_ACC: 68.0000
24-10-08 14:40:25.191 : Epoch: 14/100, Learning Rate: 0.00020 | Training, AVG_Loss 0.6083, AVG_ACC: 65.3000 | Validating, AVG_Loss 0.5428, AVG_ACC: 71.1429
24-10-08 14:43:27.733 : Epoch: 16/100, Learning Rate: 0.00020 | Training, AVG_Loss 0.5747, AVG_ACC: 68.0833 | Validating, AVG_Loss 0.5669, AVG_ACC: 71.5000
24-10-08 14:46:30.446 : Epoch: 18/100, Learning Rate: 0.00020 | Training, AVG_Loss 0.5109, AVG_ACC: 73.5000 | Validating, AVG_Loss 0.4987, AVG_ACC: 76.0714
24-10-08 14:49:33.630 : Epoch: 20/100, Learning Rate: 0.00020 | Training, AVG_Loss 0.4661, AVG_ACC: 76.5833 | Validating, AVG_Loss 0.5518, AVG_ACC: 74.2143
24-10-08 14:52:37.205 : Epoch: 22/100, Learning Rate: 0.00020 | Training, AVG_Loss 0.4557, AVG_ACC: 76.5500 | Validating, AVG_Loss 0.6434, AVG_ACC: 73.8571
24-10-08 14:55:41.985 : Epoch: 24/100, Learning Rate: 0.00020 | Training, AVG_Loss 0.5443, AVG_ACC: 68.9333 | Validating, AVG_Loss 0.5999, AVG_ACC: 62.5714
24-10-08 14:58:41.931 : Epoch: 26/100, Learning Rate: 0.00020 | Training, AVG_Loss 0.5640, AVG_ACC: 67.4500 | Validating, AVG_Loss 0.4881, AVG_ACC: 76.5000
24-10-08 15:01:42.682 : Epoch: 28/100, Learning Rate: 0.00020 | Training, AVG_Loss 0.4840, AVG_ACC: 73.0833 | Validating, AVG_Loss 0.6370, AVG_ACC: 76.0000
24-10-08 15:04:43.888 : Epoch: 30/100, Learning Rate: 0.00010 | Training, AVG_Loss 0.4463, AVG_ACC: 76.6167 | Validating, AVG_Loss 1.6238, AVG_ACC: 64.9286
24-10-08 15:07:48.455 : Epoch: 32/100, Learning Rate: 0.00010 | Training, AVG_Loss 0.3596, AVG_ACC: 81.6833 | Validating, AVG_Loss 0.6106, AVG_ACC: 76.9286
24-10-08 15:10:52.245 : Epoch: 34/100, Learning Rate: 0.00010 | Training, AVG_Loss 0.3476, AVG_ACC: 81.8333 | Validating, AVG_Loss 0.6175, AVG_ACC: 76.3571
24-10-08 15:13:57.319 : Epoch: 36/100, Learning Rate: 0.00010 | Training, AVG_Loss 0.3416, AVG_ACC: 82.6000 | Validating, AVG_Loss 0.9928, AVG_ACC: 72.2857
24-10-08 15:17:01.856 : Epoch: 38/100, Learning Rate: 0.00010 | Training, AVG_Loss 0.3561, AVG_ACC: 81.8667 | Validating, AVG_Loss 0.7695, AVG_ACC: 74.9286
24-10-08 15:20:03.545 : Epoch: 40/100, Learning Rate: 0.00010 | Training, AVG_Loss 0.3263, AVG_ACC: 82.9167 | Validating, AVG_Loss 0.5684, AVG_ACC: 77.8571
24-10-08 15:23:03.751 : Epoch: 42/100, Learning Rate: 0.00010 | Training, AVG_Loss 0.3293, AVG_ACC: 83.0333 | Validating, AVG_Loss 0.7992, AVG_ACC: 74.5000
24-10-08 15:26:04.358 : Epoch: 44/100, Learning Rate: 0.00010 | Training, AVG_Loss 0.3355, AVG_ACC: 82.9000 | Validating, AVG_Loss 0.6067, AVG_ACC: 66.2143
24-10-08 15:29:04.807 : Epoch: 46/100, Learning Rate: 0.00010 | Training, AVG_Loss 0.3233, AVG_ACC: 83.5500 | Validating, AVG_Loss 0.5986, AVG_ACC: 77.0714
24-10-08 15:32:08.801 : Epoch: 48/100, Learning Rate: 0.00010 | Training, AVG_Loss 0.3113, AVG_ACC: 83.7667 | Validating, AVG_Loss 0.6402, AVG_ACC: 77.5000
24-10-08 15:35:09.389 : Epoch: 50/100, Learning Rate: 0.00010 | Training, AVG_Loss 0.3130, AVG_ACC: 83.6667 | Validating, AVG_Loss 0.6754, AVG_ACC: 76.9286
24-10-08 15:38:13.993 : Epoch: 52/100, Learning Rate: 0.00010 | Training, AVG_Loss 0.3139, AVG_ACC: 83.8333 | Validating, AVG_Loss 1.3620, AVG_ACC: 71.3571
24-10-08 15:41:15.558 : Epoch: 54/100, Learning Rate: 0.00010 | Training, AVG_Loss 0.3065, AVG_ACC: 83.8333 | Validating, AVG_Loss 1.6721, AVG_ACC: 71.1429
24-10-08 15:44:16.709 : Epoch: 56/100, Learning Rate: 0.00010 | Training, AVG_Loss 0.3006, AVG_ACC: 84.3333 | Validating, AVG_Loss 0.8490, AVG_ACC: 75.0714
24-10-08 15:47:18.426 : Epoch: 58/100, Learning Rate: 0.00010 | Training, AVG_Loss 0.3023, AVG_ACC: 84.3000 | Validating, AVG_Loss 0.9137, AVG_ACC: 74.3571
24-10-08 15:50:20.487 : Epoch: 60/100, Learning Rate: 0.00005 | Training, AVG_Loss 0.2857, AVG_ACC: 84.7333 | Validating, AVG_Loss 0.8505, AVG_ACC: 74.7143
24-10-08 15:53:21.338 : Epoch: 62/100, Learning Rate: 0.00005 | Training, AVG_Loss 0.2793, AVG_ACC: 85.0333 | Validating, AVG_Loss 1.0839, AVG_ACC: 73.9286
24-10-08 15:56:19.903 : Epoch: 64/100, Learning Rate: 0.00005 | Training, AVG_Loss 0.2797, AVG_ACC: 85.2500 | Validating, AVG_Loss 0.7568, AVG_ACC: 77.2143
24-10-08 15:59:19.734 : Epoch: 66/100, Learning Rate: 0.00005 | Training, AVG_Loss 0.2690, AVG_ACC: 85.4000 | Validating, AVG_Loss 1.0327, AVG_ACC: 74.3571
24-10-08 16:02:25.565 : Epoch: 68/100, Learning Rate: 0.00005 | Training, AVG_Loss 0.2683, AVG_ACC: 85.4833 | Validating, AVG_Loss 0.7829, AVG_ACC: 77.5714
24-10-08 16:05:31.574 : Epoch: 70/100, Learning Rate: 0.00005 | Training, AVG_Loss 0.2734, AVG_ACC: 85.4333 | Validating, AVG_Loss 0.8920, AVG_ACC: 75.8571
24-10-08 16:08:36.765 : Epoch: 72/100, Learning Rate: 0.00005 | Training, AVG_Loss 0.2658, AVG_ACC: 85.6000 | Validating, AVG_Loss 0.8026, AVG_ACC: 78.3571
24-10-08 16:11:38.139 : Epoch: 74/100, Learning Rate: 0.00005 | Training, AVG_Loss 0.2664, AVG_ACC: 85.4667 | Validating, AVG_Loss 1.5076, AVG_ACC: 71.4286
24-10-08 16:14:40.667 : Epoch: 76/100, Learning Rate: 0.00005 | Training, AVG_Loss 0.2564, AVG_ACC: 85.8667 | Validating, AVG_Loss 0.9595, AVG_ACC: 75.5000
24-10-08 16:17:44.960 : Epoch: 78/100, Learning Rate: 0.00005 | Training, AVG_Loss 0.2635, AVG_ACC: 85.8833 | Validating, AVG_Loss 1.2341, AVG_ACC: 73.5714
24-10-08 16:20:48.175 : Epoch: 80/100, Learning Rate: 0.00005 | Training, AVG_Loss 0.2607, AVG_ACC: 86.0833 | Validating, AVG_Loss 1.2123, AVG_ACC: 74.4286
24-10-08 16:23:52.406 : Epoch: 82/100, Learning Rate: 0.00005 | Training, AVG_Loss 0.2574, AVG_ACC: 86.0500 | Validating, AVG_Loss 1.1657, AVG_ACC: 75.0714
24-10-08 16:26:54.532 : Epoch: 84/100, Learning Rate: 0.00005 | Training, AVG_Loss 0.2523, AVG_ACC: 85.9500 | Validating, AVG_Loss 1.5472, AVG_ACC: 73.2857
24-10-08 16:29:56.942 : Epoch: 86/100, Learning Rate: 0.00005 | Training, AVG_Loss 0.2496, AVG_ACC: 86.4167 | Validating, AVG_Loss 1.3173, AVG_ACC: 75.2857
24-10-08 16:33:01.218 : Epoch: 88/100, Learning Rate: 0.00005 | Training, AVG_Loss 0.2494, AVG_ACC: 86.2167 | Validating, AVG_Loss 1.1274, AVG_ACC: 74.2143
24-10-08 16:36:05.070 : Epoch: 90/100, Learning Rate: 0.00003 | Training, AVG_Loss 0.2468, AVG_ACC: 86.5000 | Validating, AVG_Loss 1.1696, AVG_ACC: 76.1429
24-10-08 16:39:07.117 : Epoch: 92/100, Learning Rate: 0.00003 | Training, AVG_Loss 0.2381, AVG_ACC: 86.7500 | Validating, AVG_Loss 1.2852, AVG_ACC: 76.4286
24-10-08 16:42:11.635 : Epoch: 94/100, Learning Rate: 0.00003 | Training, AVG_Loss 0.2355, AVG_ACC: 87.0000 | Validating, AVG_Loss 1.2966, AVG_ACC: 76.0714
24-10-08 16:45:16.004 : Epoch: 96/100, Learning Rate: 0.00003 | Training, AVG_Loss 0.2366, AVG_ACC: 86.8500 | Validating, AVG_Loss 1.3695, AVG_ACC: 75.4286
24-10-08 16:48:21.142 : Epoch: 98/100, Learning Rate: 0.00003 | Training, AVG_Loss 0.2314, AVG_ACC: 87.2000 | Validating, AVG_Loss 1.3694, AVG_ACC: 75.6429
24-10-08 16:51:28.133 : Epoch: 100/100, Learning Rate: 0.00003 | Training, AVG_Loss 0.2339, AVG_ACC: 87.0833 | Validating, AVG_Loss 1.4751, AVG_ACC: 75.2143
"""

# --- Updated Regular Expression ---
# This pattern matches:
#   - "Epoch: <number>/100,"
#   - "Learning Rate: <number>" (ignored),
#   - " | Training, AVG_Loss <number>, AVG_ACC: <number>"
#   - " | Validating, AVG_Loss <number>, AVG_ACC: <number>"
pattern = (
    r"Epoch:\s*(\d+)/100,\s*Learning Rate:\s*[\d\.]+\s*\|\s*"
    r"Training,\s*AVG_Loss\s*[\d\.]+,\s*AVG_ACC:\s*([\d\.]+)\s*\|\s*"
    r"Validating,\s*AVG_Loss\s*[\d\.]+,\s*AVG_ACC:\s*([\d\.]+)"
)

# --- Parse the Log Data ---
epochs = []
training_acc = []
validating_acc = []

for line in log_data.splitlines():
    if not line.strip():
        continue  # Skip empty lines
    match = re.search(pattern, line)
    if match:
        epoch = int(match.group(1))
        train_acc = float(match.group(2))
        valid_acc = float(match.group(3))
        epochs.append(epoch)
        training_acc.append(train_acc)
        validating_acc.append(valid_acc)
    else:
        print("No match found for line:")
        print(line)

# --- Plotting the Graphs ---
plt.figure(figsize=(10, 6))

# Chart 1: Training AVG_ACC as a blue continuous line.
plt.plot(epochs, training_acc, label='Training AVG_ACC', color='blue', linewidth=2)

# Chart 2: Validating AVG_ACC as a red dotted line.
plt.plot(epochs, validating_acc, label='Validating AVG_ACC', color='red', linestyle=':', linewidth=2)

# Set both x-axis (Epoch) and y-axis (Accuracy) limits to 1-100.
plt.xlim(1, 100)
plt.ylim(1, 100)

# Title, labels, and legend.
plt.title('Training vs. Validating Average Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Accuracy (%)')
plt.legend()
plt.grid(True)

# --- Explanation of the Graph ---
print("Graph Guide:")
print(" - Blue Continuous Line: Represents the Training AVG_ACC (accuracy on the training dataset) at each epoch.")
print(" - Red Dotted Line: Represents the Validating AVG_ACC (accuracy on the validation dataset) at each epoch.")
print("Both the x-axis (Epoch) and y-axis (Accuracy %) range from 1 to 100.")

# Display the plot.
plt.show()
