import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def load_g2o(file_path):
    poses = []
    landmarks = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("VERTEX_SE2"):
                parts = line.strip().split()
                x = float(parts[2])
                y = float(parts[3])
                poses.append((x, y))
            elif line.startswith("VERTEX_XY"):
                parts = line.strip().split()
                x = float(parts[2])
                y = float(parts[3])
                landmarks.append((x, y))
    return poses, landmarks

def plot_trajectory_and_landmarks(poses, landmarks, output_file="trajectory.png", font_size=14):
    # Set global font size
    plt.rcParams.update({'font.size': font_size})

    xs = [p[0] for p in poses]
    ys = [p[1] for p in poses]

    plt.figure(figsize=(10, 10), dpi=300)
    plt.plot(xs, ys, 'b-', linewidth=2, label="Trajectory")

    if landmarks:
        lx = [l[0] for l in landmarks]
        ly = [l[1] for l in landmarks]
        plt.scatter(lx, ly, c='red', s=30, label="Landmarks", zorder=5)

    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_file, dpi=300)
    print(f"Saved high-resolution image to {output_file}")

def load_se3_quat_poses(file_path):
    poses = []
    landmarks = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("VERTEX_SE3:QUAT"):
                parts = line.strip().split()
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                poses.append((x, y, z))
            elif line.startswith("VERTEX_XYZ"):  # optional landmark format
                parts = line.strip().split()
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                landmarks.append((x, y, z))
    return poses, landmarks

def plot_trajectory_and_landmarks_3d(poses, landmarks=None, output_file="trajectory_3d.png", font_size=14):
    plt.rcParams.update({'font.size': font_size})

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    xs = [p[0] for p in poses]
    ys = [p[1] for p in poses]
    zs = [p[2] for p in poses]
    ax.plot(xs, ys, zs, 'b-', linewidth=2, label="Trajectory")

    if landmarks:
        lx = [l[0] for l in landmarks]
        ly = [l[1] for l in landmarks]
        lz = [l[2] for l in landmarks]
        ax.scatter(lx, ly, lz, c='red', s=30, label="Landmarks", zorder=5)

    ax.set_xlabel("X (meters)", labelpad=15)
    ax.set_ylabel("Y (meters)", labelpad=15)
    ax.set_zlabel("Z (meters)", labelpad=15)
    ax.legend() # Make axes equal
    max_range = max(
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs)
    ) / 2.0

    mid_x = (max(xs) + min(xs)) / 2.0
    mid_y = (max(ys) + min(ys)) / 2.0
    mid_z = (max(zs) + min(zs)) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved high-resolution 3D image to {output_file}")

if __name__ == "__main__":
    g2o_file = "results/CPGO/parking-garage_results.g2o"  # Replace with your actual .g2o path
    poses, landmarks = load_se3_quat_poses(g2o_file)
    plot_trajectory_and_landmarks_3d(poses, landmarks, "parking-garage.png", font_size=25)





