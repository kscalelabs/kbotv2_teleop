import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('vr_teleop/dev/vr_controller_data.csv')

# Convert the timestamp to a more readable format if necessary
# data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

# Plot position data
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(data['timestamp'], data['pos_x'], label='Position X')
plt.plot(data['timestamp'], data['pos_y'], label='Position Y')
plt.plot(data['timestamp'], data['pos_z'], label='Position Z')
plt.title('VR Controller Position Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Position')
plt.legend()

# Plot Euler angles data
plt.subplot(2, 1, 2)
plt.plot(data['timestamp'], data['rot_x'], label='Euler Angle X')
plt.plot(data['timestamp'], data['rot_y'], label='Euler Angle Y')
plt.plot(data['timestamp'], data['rot_z'], label='Euler Angle Z')
plt.title('VR Controller Euler Angles Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Euler Angles (degrees)')
plt.legend()

plt.tight_layout()

# Save the plot to a file
plt.savefig('vr_controller_plot.png')
