import numpy as np
import matplotlib.pyplot as plt

def load_tum_no_timestamp(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                tx, ty, tz = map(float, parts[1:4])
                data.append([tx, ty, tz])
    return np.array(data)

# === Change these paths to your TUM files ===
file1 = 'results/CIN/inp/cert/MH3500_incincremental_result_1.tum'
file2 = 'results/CIN/inp/isam/ISAM_poses2.tum'

# Load data
traj1 = load_tum_no_timestamp(file1)
traj2 = load_tum_no_timestamp(file2)

# Plotting
plt.figure(figsize=(8, 8))
plt.plot(traj1[:, 0], traj1[:, 1], 'r-', label='Trajectory 1')
plt.plot(traj2[:, 0], traj2[:, 1], 'b-', label='Trajectory 2')

plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.axis('equal')
plt.title('TUM Trajectories')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
