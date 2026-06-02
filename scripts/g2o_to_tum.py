import math

# ─── User parameters ───────────────────────────────────────────────────────────
input_path  = "data/pgo/input_M3500.g2o"      # your .g2o file
output_path = "M3500gt.tum"     # desired TUM file
# ──────────────────────────────────────────────────────────────────────────────

with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
    for line in fin:
        if not line.startswith("VERTEX_SE2"):
            continue
        parts = line.split()
        idx   = float(parts[1])             # use as timestamp
        x, y  = float(parts[2]), float(parts[3])
        theta = float(parts[4])
        # compute quaternion for yaw = theta
        qz = math.sin(theta / 2.0)
        qw = math.cos(theta / 2.0)
        # write: time x y z qx qy qz qw
        fout.write(f"{idx:.6f} {x:.6f} {y:.6f} 0.000000 "
                   f"0.000000 0.000000 {qz:.6f} {qw:.6f}\n")
print(f"Wrote TUM trajectory to {output_path}")