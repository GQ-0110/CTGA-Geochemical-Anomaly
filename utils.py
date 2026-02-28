import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import scipy.io as sio
import os
from scipy.ndimage import minimum_filter, maximum_filter

def plot_roc_curve(y, pred, title):
    # ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    y = np.where(y == 3, 1, y)
    fpr, tpr, threshold = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    print('roc_auc:', roc_auc)



def save_model(model, optimizer, args, epoch, save_file):
    print('==> Saving...')
    state = {
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def enhance_anomaly_open_close(hsi, win_small=3, win_large=5, mode='average'):

    H, W, B = hsi.shape

    enh_small = np.zeros((H, W, B))
    enh_large = np.zeros((H, W, B))

    for b in range(B):
        band = hsi[:, :, b]
        open_s = maximum_filter(minimum_filter(band, size=win_small), size=win_small)
        open_l = maximum_filter(minimum_filter(band, size=win_large), size=win_large)
        close_s = minimum_filter(maximum_filter(band, size=win_small), size=win_small)
        close_l = minimum_filter(maximum_filter(band, size=win_large), size=win_large)

        enh_small[:, :, b] = close_s - open_s
        enh_large[:, :, b] = close_l - open_l

    if mode == 'max':
        enh_small_fused = np.max(enh_small, axis=2)
    else:  # average
        enh_small_fused = np.mean(enh_small, axis=2)

    enh_small_fused = (enh_small_fused - enh_small_fused.min()) / (enh_small_fused.max() - enh_small_fused.min())
    return enh_small_fused

def loadData(name):
    data_path = os.path.join(os.getcwd(), 'data')
    if name == 'Sandiego2':
        data = sio.loadmat(os.path.join(data_path, 'Sandiego2', 'sandiego.mat'))['data']
        data1 = sio.loadmat(os.path.join(data_path, 'Sandiego2', 'sandiego.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, 'Sandiego2','groundtruth.mat'))['gt']
    data = np.log(data + 1e-8)
    data1 = np.log(data1 + 1e-8)
    return data1,data, labels
