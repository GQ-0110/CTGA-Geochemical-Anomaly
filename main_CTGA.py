import os
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import moco.builder
import moco.loader
from utils import loadData, save_model
from vit_pytorch import Transformer as ViT

DATASET = "zs"
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-6
PRINT_FREQ = 30
SAVE_FREQ = 50

# MoCo params
MOCO_DIM = 128
MOCO_MLP_DIM = 256
MOCO_M = 0.99
MOCO_K = 4096
MOCO_T = 0.07

# Output
TRIAL_ID = "0"
SAVE_ROOT = "./save"


# -------------------------
# Utils
# -------------------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class MaskTransform:
    """Simple masking augmentation for one view."""
    def __init__(self, mask_ratio=0.5):
        self.mask_ratio = mask_ratio

    def __call__(self, img: Image.Image):
        arr = np.array(img)
        h, w = arr.shape
        mask = np.random.random((h, w)) > self.mask_ratio
        arr[~mask] = 0
        return arr


def build_save_folder():
    model_path = os.path.join(SAVE_ROOT, f"{DATASET}_models")
    model_name = f"{DATASET}_lr_{LEARNING_RATE}_bsz_{BATCH_SIZE}_temp_{MOCO_T}_trial_{TRIAL_ID}"
    save_folder = os.path.join(model_path, model_name)
    os.makedirs(save_folder, exist_ok=True)
    return save_folder


def set_model(device):
    vit = ViT(
        band_size=11,
        patch_size=3,
        embed_dim=128,
        depth=4,
        num_heads=4,
        num_classes=2,
    )
    vit.head = nn.Identity()

    model = moco.builder.MoCo_ViT(
        vit,
        dim=MOCO_DIM,
        mlp_dim=MOCO_MLP_DIM,
        T=MOCO_T,
        K=MOCO_K,
        m=MOCO_M,
    ).to(device)

    if device.type == "cuda":
        cudnn.benchmark = True

    return model


def set_optimizer(model):
    return optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )


def set_loader():
    data1, data2, _ = loadData(DATASET)

    data1 = np.transpose(data1, (2, 0, 1))  # (bands, H, W)
    data2 = np.transpose(data2, (2, 0, 1))  # (bands, H, W)
    bands, H, W = data1.shape

    mask_aug = MaskTransform(mask_ratio=0.5)

    aug_data2 = []
    for i in range(bands):
        img = Image.fromarray(data2[i, :, :].astype(np.float32))
        aug = mask_aug(img)
        aug_data2.append(aug[None, :])
    aug_data2 = np.vstack(aug_data2)  # (bands, H, W)

    x1 = np.transpose(data1, (1, 2, 0)).reshape(H * W, bands)
    x2 = np.transpose(aug_data2, (1, 2, 0)).reshape(H * W, bands)

    dataset = moco.loader.TwoCropsTransform(x1, x2)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    return loader


def train_one_epoch(train_loader, model, optimizer, epoch, device):
    model.train()
    losses = AverageMeter()

    for i, (x1, x2) in enumerate(train_loader):
        x1 = x1.to(device)
        x2 = x2.to(device)

        loss, _, _ = model(im_q=x1, im_k=x2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), n=x1.size(0))

        if (i + 1) % PRINT_FREQ == 0:
            print(f"Train: [{epoch}][{i+1}/{len(train_loader)}]\t"
                  f"loss {losses.val:.3f} ({losses.avg:.3f})")

    return losses.avg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_folder = build_save_folder()

    train_loader = set_loader()
    model = set_model(device)
    optimizer = set_optimizer(model)

    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        _ = train_one_epoch(train_loader, model, optimizer, epoch, device)

        if epoch % SAVE_FREQ == 0:
            ckpt_path = os.path.join(save_folder, f"ckpt_epoch_{epoch}.pth")
            save_model(model, optimizer, None, epoch, ckpt_path)

    # save last
    ckpt_path = os.path.join(save_folder, "last.pth")
    save_model(model, optimizer, None, EPOCHS, ckpt_path)

    print("Total time (s):", time.time() - start)
    print("Saved to:", save_folder)


if __name__ == "__main__":
    main()