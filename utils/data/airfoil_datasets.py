import numpy as np
import scipy.io as sio
import torch
import torchvision.transforms as transforms
from scipy.interpolate import griddata
from torch.utils.data import Dataset


def transform():
    input_trans = transforms.Normalize(mean=[28.871367175417486, -0.05779870714590324, 0.02325439453125],
                                       std=[7.196371834502616, 3.0105339279944676, 0.1507104099464764])
    target_trans = transforms.Normalize(mean=[-18.628928978060046, 29.066497993715878, -0.08813911066792186],
                                        std=[70.14272987786762, 7.573816244044205, 3.2647237850144504])
    # target_trans = transforms.Normalize(mean=[29.066497993715878, -0.08813911066792186],
    #                                     std=[7.573816244044205, 3.2647237850144504])
    return input_trans, target_trans


class PointDataset(Dataset):
    def __init__(self, root, split, opt):
        self.root = root
        self.input_trans, self.target_trans = transform()
        input_path = "/mnt/zyy/3D/input_256.mat"
        target_path = "/mnt/zyy/3D/target_256.mat"
        input_mats = sio.loadmat(input_path)
        target_mats = sio.loadmat(target_path)
        # input_sst = np.array(input_mats['data'])
        mask = 1 - np.array(input_mats['data'])[0, 2, ...]
        target_sst = np.array(target_mats['data'])
        if split == 'train':
            # self.input = input_sst[:opt['labeled_num']]
            self.data = target_sst[:opt['labeled_num']]
        if split == 'ul_train':
            # self.input = input_sst[800: 1800]
            self.data = target_sst[800: 1600]
        if split == 'test':
            # self.input = input_sst[1800: 2000]
            self.data = target_sst[1600: 2000]
        self.data = self.target_trans(torch.tensor(self.data))
        B, N, H, W = self.data.shape
        self.target = self.data.clone().reshape(B, N, H * W)
        self.mask = torch.tensor(mask).reshape(-1, H * W)
        self.target = self.target * self.mask
        self.input = self.data[:, :, [64, 65, 65, 110, 110, 192], [128, 126, 130, 121, 139, 128]]

    def __getitem__(self, index):
        input = self.input[index, ...]
        target = self.target[index, ...]
        return input, target, self.mask

    def __len__(self):
        return len(self.data)


def standardization(data):
    mean = np.array([-18.628928978060046, 29.066497993715878, -0.08813911066792186])
    std = np.array([70.14272987786762, 7.573816244044205, 3.2647237850144504])
    for i in range(data.shape[1]):
        data[:, i, ...] = (data[:, i, ...] - mean[i]) / std[i]
    return data


class InterpolDataset(Dataset):
    def __init__(self, root, split, opt):
        self.root = root
        # self.input_trans, self.target_trans = transform()
        input_path = "/mnt/zyy/3D/input_256.mat"
        target_path = "/mnt/zyy/3D/target_256.mat"
        input_mats = sio.loadmat(input_path)
        target_mats = sio.loadmat(target_path)
        mask = 1 - np.array(input_mats['data'])[0, 2, ...]
        target_sst = np.array(target_mats['data'])
        target_sst = standardization(target_sst)
        target_sst = target_sst * mask

        positions = np.array([[64, 128], [65, 126], [65, 130], [110, 121], [110, 139], [192, 128]])
        _, _, h, w = target_sst.shape
        x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
        x_coor, y_coor = np.meshgrid(x_coor, y_coor)
        sparse_locations_ex = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            sparse_locations_ex[i, 0] = x_coor[positions[i, 0], positions[i, 1]]
            sparse_locations_ex[i, 1] = y_coor[positions[i, 0], positions[i, 1]]

        if split == 'train':
            target_sst = target_sst[:opt['labeled_num']]
        if split == 'ul_train':
            target_sst = target_sst[800: 1600]
        if split == 'test':
            target_sst = target_sst[1600: 2000]

        sparse_data_all = []
        print("processing data...")
        for j in range(target_sst.shape[0]):
            sparse_data = []
            for i in range(positions.shape[0]):
                sparse_data.append(target_sst[j, :, positions[i, 0], :][:, positions[i, 1]].reshape(-1, 1))
            sparse_data = np.concatenate(sparse_data, axis=-1)

            sparse_datas = []
            for i in range(sparse_data.shape[0]):
                input_1 = griddata(sparse_locations_ex, sparse_data[i, ...], (x_coor, y_coor), method='nearest')
                sparse_datas.append(np.expand_dims(input_1, axis=0))
            sparse_datas = np.concatenate(sparse_datas, axis=0)
            sparse_data_all.append(np.expand_dims(sparse_datas, axis=0))
        sparse_data_all = np.concatenate(sparse_data_all, axis=0)
        sparse_data_all = sparse_data_all * mask

        self.input = torch.tensor(sparse_data_all)
        self.target = torch.tensor(target_sst)
        self.mask = torch.tensor(mask)

    def __getitem__(self, index):
        input = self.input[index, ...]
        target = self.target[index, ...]
        return input, target, self.mask

    def __len__(self):
        return len(self.input)
