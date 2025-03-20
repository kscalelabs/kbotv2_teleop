import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('logs/ik_solving.csv')

# Create a figure with multiple subplots (now just 2 instead of 3)
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot 1: Joint angles vs iteration
axs[0].set_title('Joint Angles vs Iteration')
for col in ['right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 
            'right_elbow', 'right_wrist']:
    if col in df.columns:
        axs[0].plot(df['iteration'], df[col], marker='.', label=col)
axs[0].set_ylabel('Angle (rad)')
axs[0].legend()
axs[0].grid(True)

axs[1].set_title('Filtered Errors vs Iteration (Excluding Large Initial Errors)')
if 'pos_error' in df.columns:
    # Filter out iterations with very large position errors
    pos_error_threshold = np.percentile(df['pos_error'], 75) * 1.5  # Using 1.5x the 75th percentile as threshold
    filtered_pos = df[df['pos_error'] < pos_error_threshold]
    axs[1].plot(filtered_pos['iteration'], filtered_pos['pos_error'], marker='.', label='Position Error')
if 'rot_error' in df.columns:
    # Filter out iterations with very large rotation errors
    rot_error_threshold = np.percentile(df['rot_error'], 75) * 1.5  # Using 1.5x the 75th percentile as threshold
    filtered_rot = df[df['rot_error'] < rot_error_threshold]
    axs[1].plot(filtered_rot['iteration'], filtered_rot['rot_error'], marker='.', label='Rotation Error')

# Get final error values
final_pos_error = df['pos_error'].iloc[-1] if 'pos_error' in df.columns else None
final_rot_error = df['rot_error'].iloc[-1] if 'rot_error' in df.columns else None

# Update the title to include final error values
title = 'Filtered Errors vs Iteration (Excluding Large Initial Errors)\n'
if final_pos_error is not None:
    title += f'Final Pos Error: {final_pos_error:.6f}, '
if final_rot_error is not None:
    title += f'Final Rot Error: {final_rot_error:.6f}'
axs[1].set_title(title)

# Add a horizontal line at y=0
axs[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)

axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Error (filtered)')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
# plt.savefig('ik_solving_plot.png')
plt.show()
