import numpy as np

# -----------------------------------------------------------------------------
# 1) Set your file names and max number of poses to compare here:
# -----------------------------------------------------------------------------
gt_path  = "results/CIN/MH3500_Inc_batch.txt"               # full “reference” trajectory
#est_path = "results/CIN/inp/cert/MH3500_incincremental_result_8.tum"  # “to be evaluated”
max_poses = 200  # only compare up to this many estimated poses

for i in range(27):
    try:
        #est_path = f"results/CIN/inp/cert/MH3500_incincremental_result_{i}.tum"
        est_path = f"results/CIN/inp/isam/ISAM_poses{i}.tum"
        # -----------------------------------------------------------------------------
        # 2) Read a TUM file into {timestamp: (position, quaternion)} 
        # -----------------------------------------------------------------------------
        def read_tum(path):
            data = {}
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    t, x, y, z, qx, qy, qz, qw = map(float, line.split())
                    data[t] = (np.array([x, y, z]), np.array([qx, qy, qz, qw]))
            return data

        gt  = read_tum(gt_path)
        est = read_tum(est_path)

        # -----------------------------------------------------------------------------
        # 3) Build matched lists, but stop after max_poses estimated matches
        # -----------------------------------------------------------------------------
        common_gt  = []
        common_est = []

        for t_est in sorted(est.keys()):
            if len(common_est) >= max_poses:
                break
            # find closest ground‐truth time
            t_gt = min(gt.keys(), key=lambda t: abs(t - t_est))
            if abs(t_gt - t_est) > 0.02:  # skip if >20 ms apart
                continue
            p_g, q_g = gt[t_gt]
            p_e, q_e = est[t_est]
            common_gt.append((p_g, q_g))
            common_est.append((p_e, q_e))

        if not common_est:
            raise RuntimeError(f"No matches found within 20 ms for first {max_poses} estimates!")

        # -----------------------------------------------------------------------------
        # 4) Stack into arrays for Umeyama and error computation
        # -----------------------------------------------------------------------------
        P_gt = np.vstack([p for p, _ in common_gt]).T  # shape 3×N
        P_es = np.vstack([p for p, _ in common_est]).T
        Q_gt = [q for _, q in common_gt]
        Q_es = [q for _, q in common_est]

        # -----------------------------------------------------------------------------
        # 5) Umeyama alignment (no scale by default)
        # -----------------------------------------------------------------------------
        def umeyama(A, B, with_scale=False):
            assert A.shape == B.shape and A.shape[0] == 3
            N = A.shape[1]
            mu_A = A.mean(axis=1, keepdims=True)
            mu_B = B.mean(axis=1, keepdims=True)
            AA = A - mu_A
            BB = B - mu_B
            Sigma = BB @ AA.T / N
            U, D, VT = np.linalg.svd(Sigma)
            S = np.eye(3)
            if np.linalg.det(U @ VT) < 0:
                S[2, 2] = -1
            R = U @ S @ VT
            s = (np.trace(D @ S) / ((AA**2).sum() / N)) if with_scale else 1.0
            t = mu_B[:,0] - s * R @ mu_A[:,0]
            return R, t, s

        R, t, s = umeyama(P_es, P_gt, with_scale=False)
        P_es_aligned = (s * R @ P_es) + t[:,None]

        # -----------------------------------------------------------------------------
        # 6) Compute translation RMSE
        # -----------------------------------------------------------------------------
        trans_errs = np.linalg.norm(P_es_aligned - P_gt, axis=0)
        rmse_trans = np.sqrt(np.mean(trans_errs**2))

        # -----------------------------------------------------------------------------
        # 7) Compute rotation RMSE
        # -----------------------------------------------------------------------------
        def quat_conj(q):
            return np.array([-q[0], -q[1], -q[2], q[3]])

        def quat_mul(a, b):
            x1,y1,z1,w1 = a
            x2,y2,z2,w2 = b
            return np.array([
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
                w1*w2 - x1*x2 - y1*y2 - z1*z2
            ])

        rot_errs = []
        for q_e, q_g in zip(Q_es, Q_gt):
            q_err = quat_mul(q_g, quat_conj(q_e))
            angle = 2 * np.arccos(np.clip(abs(q_err[3]), -1.0, 1.0))
            rot_errs.append(angle)

        rmse_rot = np.sqrt(np.mean(np.array(rot_errs)**2))

        print(f"Compared {len(common_est)} poses (max requested = {max_poses})")
        print(f"Translation RMSE: {rmse_trans:.4f} m")
        print(f"Rotation RMSE:    {np.degrees(rmse_rot):.4f}°")
        max_poses += 200
    except:
        continue

