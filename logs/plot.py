import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('logs/ik_solving.csv')

# Create a figure with multiple subplots (now just 2 instead of 3)
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot 1: Joint angles vs iteration
axs[0].set_title('Joint Angles vs Iteration')

# Define joint limits from the provided values
right_arm_min_limits = np.array([
    -2.617994,  # right_shoulder_pitch_03
    -0.488692,  # right_shoulder_roll_03
    -1.745329,  # right_shoulder_yaw_02
     0.0,       # right_elbow_02
    -1.745329   # right_wrist_02
])

right_arm_max_limits = np.array([
     2.094395,  # right_shoulder_pitch_03
     1.658063,  # right_shoulder_roll_03
     1.745329,  # right_shoulder_yaw_02
     2.530727,  # right_elbow_02
     1.745329   # right_wrist_02
])

left_arm_min_limits = np.array([
    -2.094395,  # left_shoulder_pitch_03
    -1.658063,  # left_shoulder_roll_03
    -1.745329,  # left_shoulder_yaw_02
    -2.530727,  # left_elbow_02 (negative range for left arm)
    -1.745329   # left_wrist_02
])

left_arm_max_limits = np.array([
     2.617994,  # left_shoulder_pitch_03
     0.488692,  # left_shoulder_roll_03
     1.745329,  # left_shoulder_yaw_02
     0.0,       # left_elbow_02 (negative range for left arm)
     1.745329   # left_wrist_02
])

# Combined joint limits dictionary
joint_limits = {
    'right_shoulder_pitch': (right_arm_min_limits[0], right_arm_max_limits[0]),
    'right_shoulder_roll': (right_arm_min_limits[1], right_arm_max_limits[1]),
    'right_shoulder_yaw': (right_arm_min_limits[2], right_arm_max_limits[2]),
    'right_elbow': (right_arm_min_limits[3], right_arm_max_limits[3]),
    'right_wrist': (right_arm_min_limits[4], right_arm_max_limits[4]),
    
    'left_shoulder_pitch': (left_arm_min_limits[0], left_arm_max_limits[0]),
    'left_shoulder_roll': (left_arm_min_limits[1], left_arm_max_limits[1]),
    'left_shoulder_yaw': (left_arm_min_limits[2], left_arm_max_limits[2]),
    'left_elbow': (left_arm_min_limits[3], left_arm_max_limits[3]),
    'left_wrist': (left_arm_min_limits[4], left_arm_max_limits[4])
}

# Define pastel but slightly darker colors for each joint type
joint_colors = {
    'right_shoulder_pitch': '#e06666',  # pastel red
    'right_shoulder_roll': '#6aa84f',   # pastel green
    'right_shoulder_yaw': '#6d9eeb',    # pastel blue
    'right_elbow': '#9966cc',           # pastel purple
    'right_wrist': '#e69138',           # pastel orange
    
    'left_shoulder_pitch': '#cc4125',   # darker pastel red
    'left_shoulder_roll': '#4c9141',    # darker pastel green
    'left_shoulder_yaw': '#4a86e8',     # darker pastel blue
    'left_elbow': '#7e57c2',            # darker pastel purple
    'left_wrist': '#d27d2d'             # darker pastel orange
}

# Get joint columns from dataframe - looking for columns that start with "right_" or "left_"
right_joint_cols = [col for col in df.columns if col.startswith('right_') and col not in ['right_pos_error', 'right_rot_error']]
left_joint_cols = [col for col in df.columns if col.startswith('left_') and col not in ['left_pos_error', 'left_rot_error']]

# Combine all joint columns that exist in the data
joint_cols = right_joint_cols + left_joint_cols

# Plot each joint with its assigned color and add limit lines with the same color
for col in joint_cols:
    color = joint_colors.get(col, None)  # Use the predefined color or None if not specified
    
    if color:
        line, = axs[0].plot(df['iteration'], df[col], marker='.', markersize=3, label=col, color=color)
    else:
        line, = axs[0].plot(df['iteration'], df[col], marker='.', markersize=3, label=col)
        color = line.get_color()  # If no predefined color, get the auto-assigned color
    
    # Add horizontal lines for joint limits if available
    if col in joint_limits:
        lower_limit, upper_limit = joint_limits[col]
        axs[0].axhline(y=lower_limit, color=color, linestyle='--', alpha=0.5)
        axs[0].axhline(y=upper_limit, color=color, linestyle='--', alpha=0.5)

axs[0].set_ylabel('Angle (rad)')
axs[0].legend()
axs[0].grid(True)

axs[1].set_title('Filtered Errors vs Iteration (Excluding Large Initial Errors)')
if 'pos_error' in df.columns:
    # Filter out iterations with very large position errors
    pos_error_threshold = np.percentile(df['pos_error'], 75) * 1.5  # Using 1.5x the 75th percentile as threshold
    filtered_pos = df[df['pos_error'] < pos_error_threshold]
    axs[1].plot(filtered_pos['iteration'], filtered_pos['pos_error'], marker='.', markersize=3, label='Position Error')
if 'rot_error' in df.columns:
    # Filter out iterations with very large rotation errors
    rot_error_threshold = np.percentile(df['rot_error'], 75) * 1.5  # Using 1.5x the 75th percentile as threshold
    filtered_rot = df[df['rot_error'] < rot_error_threshold]
    axs[1].plot(filtered_rot['iteration'], filtered_rot['rot_error'], marker='.', markersize=3, label='Rotation Error')

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
