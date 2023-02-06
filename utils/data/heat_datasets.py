import os

import h5py
import numpy as np
import scipy.io as sio
import torch
from scipy.interpolate import griddata
from torch.utils.data import Dataset


class PointDataset(Dataset):
    def __init__(self, root, split, opt):
        self.root = root

        path = "/mnt/zyy/3D/temperature.h5"
        data = h5py.File(path, "r")
        sst = np.array(data['u'])
        if split == 'train':
            self.data = sst[list(range(opt['labeled_num'])), :]
        if split == 'ul_train':
            self.data = sst[list(range(2000, 4000)), :]
        if split == 'test':
            self.data = sst[list(range(4000, 5000)), :]
        B, N, H, W = self.data.shape
        self.data = (self.data - 298) / 50
        self.target = self.data.copy().reshape(B, N, H * W)
        self.input = self.data[:, :, 39::40, 40::40].reshape(B, N, 20)

        # mask = np.zeros_like(self.target)
        # mask[:, n_idx["incp_point"][0, :]] = 1
        # self.target_m = self.target * mask

    def __getitem__(self, index):
        input = torch.tensor(self.input[index, ...])
        target = torch.tensor(self.target[index, ...])
        # target_m = torch.tensor(self.target[index, ...])
        return input, target

    def __len__(self):
        return len(self.data)


class PointDataset1(Dataset):
    def __init__(self, root, split, opt):
        self.root = root
        self.data = []
        if split == 'train':
            data_path = "/mnt/zyy/3D/real_data_88/train"
            for i in range(opt['labeled_num']):
                data_dir = os.path.join(data_path, f"Example{i}.mat")
                data = sio.loadmat(data_dir)
                self.data.append(data[f'u'])
        if split == 'ul_train':
            data_path = "/mnt/zyy/3D/real_data_88/train"
            for i in range(600, 1200):
                data_dir = os.path.join(data_path, f"Example{i}.mat")
                data = sio.loadmat(data_dir)
                self.data.append(data[f'u'])
        if split == 'test':
            data_path = "/mnt/zyy/3D/real_data_88/test"
            for i in range(1200, 1600):
                data_dir = os.path.join(data_path, f"Example{i}.mat")
                data = sio.loadmat(data_dir)
                self.data.append(data[f'u'])
        self.data = np.array(np.expand_dims(self.data, axis=1))
        B, _, H, W = self.data.shape
        self.data = (self.data - 320) / 100
        self.target = self.data.copy().reshape(B, 1, H * W)
        idx = np.array(np.where(data["position"] == 1))
        self.input = self.data[:, :, idx[0], idx[1]]
        # self.input = self.data[:, :, [42, 8, 59, 80, 175, 121], [26, 128, 191, 32, 12, 132]]
        print(1)

        # mask = np.zeros_like(self.target)
        # mask[:, n_idx["incp_point"][0, :]] = 1
        # self.target_m = self.target * mask

    def __getitem__(self, index):
        input = torch.tensor(self.input[index, ...])
        target = torch.tensor(self.target[index, ...])
        # target_m = torch.tensor(self.target[index, ...])
        return input, target

    def __len__(self):
        return len(self.data)


class InterpolDataset(Dataset):
    def __init__(self, root, split, opt):
        self.root = root
        # self.input_trans, self.target_trans = transform()
        path = "/mnt/zyy/3D/temperature.h5"
        data = h5py.File(path, "r")
        sst = np.array(data['u'])
        sst = (sst - 298) / 50

        positions = np.array([[39, 40], [79, 40], [119, 40], [159, 40], [199, 40],
                              [39, 80], [79, 80], [119, 80], [159, 80], [199, 80],
                              [39, 120], [79, 120], [119, 120], [159, 120], [199, 120],
                              [39, 160], [79, 160], [119, 160], [159, 160], [199, 160]])
        _, _, h, w = sst.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)
        sparse_locations_ex = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            sparse_locations_ex[i, 0] = x_coor[positions[i, 0], positions[i, 1]]
            sparse_locations_ex[i, 1] = y_coor[positions[i, 0], positions[i, 1]]

        if split == 'train':
            sst = sst[:opt['labeled_num']]
        if split == 'ul_train':
            sst = sst[2000: 4000]
        if split == 'test':
            sst = sst[4000: 5000]

        sparse_data_all = []
        print("processing data...")
        for j in range(sst.shape[0]):
            sparse_data = []
            for i in range(positions.shape[0]):
                sparse_data.append(sst[j, :, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
            sparse_data = np.concatenate(sparse_data, axis=-1)

            sparse_datas = []
            for i in range(sparse_data.shape[0]):
                input_1 = griddata(sparse_locations_ex, sparse_data[i, ...], (x_coor, y_coor), method='nearest')
                sparse_datas.append(np.expand_dims(input_1, axis=0))
            sparse_datas = np.concatenate(sparse_datas, axis=0)
            sparse_data_all.append(np.expand_dims(sparse_datas, axis=0))
        sparse_data_all = np.concatenate(sparse_data_all, axis=0)

        self.input = torch.tensor(sparse_data_all)
        self.target = torch.tensor(sst)

    def __getitem__(self, index):
        input = self.input[index, ...]
        target = self.target[index, ...]
        return input, target

    def __len__(self):
        return len(self.input)
