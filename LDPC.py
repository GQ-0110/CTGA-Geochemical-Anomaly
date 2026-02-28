import numpy as np
import scipy.io as sio
from sklearn.metrics import pairwise_distances

def zscore(X, eps=1e-12):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + eps)

def ldpc_select_targets(
    data_mat_path="data.mat",
    H=90, W=90,
    win_size=9,
    stride=3,
    num_targets_per_block=3,
    top_k_global=20,
    eps=1e-6,
):
    data = sio.loadmat(data_mat_path)["data"]          # (H,W,B)
    assert data.shape[0] == H and data.shape[1] == W, "H/W mismatch with data"
    B = data.shape[2]

    X1 = data.reshape(-1, B)  # (H*W, B)
    global_std_ref = np.mean(np.std(X1, axis=0))
    dc = np.sqrt(B) * global_std_ref

    n_grid = int((H - win_size) / stride + 1)

    all_gamma_values = []
    all_positions = []

    for i_block in range(n_grid):
        for j_block in range(n_grid):
            start_row = 0 + i_block * stride
            start_col = 0 + j_block * stride
            end_row = start_row + win_size
            end_col = start_col + win_size

            sub = data[start_row:end_row, start_col:end_col, :]  # (win,win,B)
            X = sub.reshape(-1, B)

            if np.all(X == 0):
                continue

            X = zscore(X)

            D = pairwise_distances(X, X, metric="euclidean")
            rho = (D < dc).sum(axis=1) - 1

            n = X.shape[0]
            idx_max = int(np.argmax(rho))
            delta = np.zeros(n, dtype=np.float64)

            delta[idx_max] = np.max(D[idx_max, :])

            for k in range(n):
                if k == idx_max:
                    continue
                higher = np.where(rho > rho[k])[0]
                if higher.size > 0:
                    delta[k] = np.min(D[k, higher])
                else:
                    delta[k] = np.max(D[k, :])

            gamma = rho * delta

            anomaly_score = 1.0 / (rho.astype(np.float64) + eps)
            sorted_idx = np.argsort(-anomaly_score)  # descending

            target_indices = sorted_idx[:num_targets_per_block]

            for idx1d in target_indices:
                local_row = idx1d // win_size
                local_col = idx1d % win_size
                global_row = start_row + local_row
                global_col = start_col + local_col

                all_gamma_values.append(gamma[idx1d])
                all_positions.append([global_row + 1, global_col + 1])

    all_gamma_values = np.asarray(all_gamma_values, dtype=np.float64)
    all_positions = np.asarray(all_positions, dtype=np.int32)

    if all_gamma_values.size > top_k_global:
        top_idx = np.argsort(-all_gamma_values)[:top_k_global]
        all_positions = all_positions[top_idx]

    global_target_mask = np.zeros((H, W), dtype=bool)
    for r1, c1 in all_positions:
        r = r1 - 1
        c = c1 - 1
        if 0 <= r < H and 0 <= c < W:
            global_target_mask[r, c] = True

    X_all = data.reshape(-1, B)
    mask_vec = global_target_mask.reshape(-1)
    data_1 = X_all.copy()
    data_1[~mask_vec] /= 100.0
    print("data_1 shape:", data_1.shape)
    return global_target_mask, data_1, all_positions


if __name__ == "__main__":
    mask, data_1, positions = ldpc_select_targets(
        data_mat_path="data.mat",
        H=90, W=90,
        win_size=9,
        stride=3,
        num_targets_per_block=3,
        top_k_global=30,
    )
