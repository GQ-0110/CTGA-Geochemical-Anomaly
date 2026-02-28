import time
import numpy as np
import scipy.io as sio
import torch

from vit_pytorch import Transformer as ViT
from utils import plot_roc_curve, enhance_anomaly_open_close
from sklearn.metrics import pairwise_distances
import moco.builder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_name = "zs"
H, W = 90, 90
BAND_SIZE = 11

data_dir = f"./data/{data_name}"
data_path = f"{data_dir}/zs.mat"
gdt_path = f"{data_dir}/groundtruth.mat"
prior_path = f"{data_dir}/prior_target.mat"

ckpt_path = f"./save/{data_name}_models/{data_name}_lr_0.05_bsz_64_temp_0.07_trial_0/ckpt_epoch_50.pth"

gdt_map = sio.loadmat(gdt_path)["gt"]
gdt = gdt_map.reshape(-1)

data = sio.loadmat(data_path)["data"]
data = data.reshape(H * W, -1).astype(np.float32)

prior = sio.loadmat(prior_path)["prior_target"].T
prior = prior.astype(np.float32)

data = np.log(data + 1e-8)
mn, mx = data.min(), data.max()
data = (data - mn) / (mx - mn + 1e-12)

vit = ViT(
    band_size=BAND_SIZE,
    patch_size=3,
    embed_dim=128,
    depth=4,
    num_heads=4,
    num_classes=128,
)

model = moco.builder.MoCo_ViT(vit, dim=128, mlp_dim=256, T=0.07, K=4096, m=0.99).to(device)

ckpt = torch.load(ckpt_path, map_location="cpu")
state_dict = ckpt.get("model", ckpt)

new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict[k.replace("module.", "")] = v
model.load_state_dict(new_state_dict, strict=True)

start = time.time()
model.eval()

prior_t = torch.from_numpy(prior).to(device)
data_t = torch.from_numpy(data).to(device)

with torch.no_grad():
    prior_feat = model.base_encoder.forward_features(prior_t)
    data_feat = model.base_encoder.forward_features(data_t)

prior_feat = prior_feat.detach().cpu().numpy()
data_feat = data_feat.detach().cpu().numpy()

dist = pairwise_distances(prior_feat, data_feat, metric="euclidean").reshape(-1)

dmin, dmax = dist.min(), dist.max()
score = (dist - dmin) / (dmax - dmin + 1e-12)  # 0~1

score_map = score.reshape(H, W, order="C")
score_map = enhance_anomaly_open_close(score_map)

score_vec = score_map.reshape(-1, 1, order="C")  # (8100,1)

plot_roc_curve(gdt, score_vec, data_name)

end = time.time()
print("running time:", end - start)
