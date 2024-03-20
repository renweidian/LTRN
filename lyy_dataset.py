import glob
import os

import hdf5storage as h5
import torch
import torch.utils.data as data
import numpy as np
from scipy.io import loadmat
import h5py


class MYDataset(data.Dataset):
    def __init__(self, data_path):
        data_names = glob.glob(os.path.join(data_path, '*.mat'))
        self.keys = data_names

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5.loadmat(self.keys[index])
        hyper = np.float32(np.array(mat['HSI']))
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['RGB']))
        rgb = torch.Tensor(rgb)
        return rgb, hyper


