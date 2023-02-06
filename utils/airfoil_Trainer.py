#!coding:utf-8
import math
import os.path as osp
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import repeat

from utils.Trainer import Trainer
from utils.quantile_regression_loss import QuantileRegressionLoss
from torch.distributions.normal import Normal


class SubTrainer(Trainer):
    def __init__(self, opt, writer):
        super(SubTrainer, self).__init__(opt, writer)
        # self.model = self.init_model('MultiMLP', [6, 128, 1280, 4800, 65536])
        # self.model = self.init_model('MLP', [6, 128, 1280, 4800, 65536])
        self.model = self.init_model('UNet', 3, 6)  # 用分位数时加1
        # self.model = self.init_model('multi_UNet', 3, 3)
        # self.model = self.init_model('MT_UNet', 3, 3)
        self.optimizer, self.scheduler = self.init_training(self.model.parameters())
        self.model = self.model.to(self.device)

    def stp_loop(self, train_loader, ul_train_loader, test_loader):
        min_loss_t, min_loss_s = 1e6, 1e6
        current_iter, epoch = 0, 0
        # T model train
        self.model.train()
        print(f"-------- Training Teacher --------")
        while current_iter < self.opt['iters']:
            for batch_idx, (data, targets, mask) in enumerate(train_loader):
                current_iter += 1
                self.optimizer.zero_grad()
                data, targets = data.float().to(self.device), targets.to(self.device)
                mask = mask.to(self.device)[0, ...]

                y1, y2, y3 = self.model(data)
                out = torch.stack([y1, y2, y3], dim=0)
                out_mean = torch.mean(out, dim=0)
                out_mean = out_mean * mask

                loss = self.loss(out_mean, targets)
                loss.backward()
                self.optimizer.step()
                lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('train_T_loss', loss, current_iter)
                if (current_iter % self.opt['print_freq']) == 0:
                    eta_str = self.time_log(current_iter)
                    print_string = f"Train:[epochs:{epoch}\t iter:{current_iter}]\t lr:{lr:.3e}\t eta:{eta_str}\t"
                    print_string += f"Loss:{loss.item():.3e}"
                    print(print_string)
                if (current_iter % self.opt['test_freq']) == 0 and current_iter >= self.opt['iters'] * 0.5:
                    print(f"-------- Testing --------")
                    self.model.eval()
                    with torch.no_grad():
                        loop_loss = []
                        for batch_idx, (data, targets, _) in enumerate(test_loader):
                            data, targets = data.float().to(self.device), targets.to(self.device)

                            y1, y2, y3 = self.model(data)
                            out = torch.stack([y1, y2, y3], dim=0)
                            out_mean = torch.mean(out, dim=0)
                            out_mean = out_mean * mask

                            loss = self.loss(out_mean, targets)
                            loop_loss.append(loss.item())
                        loop_loss = np.array(loop_loss).mean()
                        self.writer.add_scalar('Test_T_loss', loop_loss, current_iter)
                        print_string = f"Test_loss:{loop_loss:.3e}"
                        print(print_string)
                    self.model.train()
                    if loop_loss < min_loss_t:
                        min_loss_t = loop_loss
                        self.save_model(self.model, "T")
                        print("Saving models.")
                    print(f">>>Min_loss:{min_loss_t:.3e}")
            epoch += 1
            self.scheduler.step()
        self.start_time = time.time()

        # S model train
        print(f"\n-------- Training Student --------")
        model_out_path = Path(self.opt['save_dir'])
        net_path = osp.join(model_out_path, f"{self.opt['name']}_T.pth")
        # net_path = "/mnt/zyy/reconstruction/experiments/airfoil_stp_l5e1_cos_mlp/model/airfoil_stp_l5e1_cos_mlp_T.pth"
        checkpoint = torch.load(net_path)
        self.model.load_state_dict(checkpoint['weight'])
        self.model.eval()
        # init student model
        self.opt['iters'] = self.opt['iters'] * 2
        self.opt['scheduler']['T_max'] = self.opt['epoch'] * 2
        net = self.init_model('UNet', 3, 3)
        net = net.to(self.device)
        self.optimizer, self.scheduler = self.init_training(net.parameters())
        net.train()
        current_iter, epoch = 0, 0
        unl_loss = torch.nn.L1Loss(reduction="none")
        while current_iter < self.opt['iters']:
            for batch_idx, (ul_data, ul_targets, mask) in enumerate(ul_train_loader):
                current_iter += 1
                self.optimizer.zero_grad()
                ul_input = ul_data.float().to(self.device)
                # ul_targets = ul_targets.to(self.device)
                mask = mask.to(self.device)[0, ...]

                with torch.no_grad():
                    y1, y2, y3 = self.model(ul_input)
                    out = torch.stack([y1, y2, y3], dim=0)
                    pseudo_mean = torch.mean(out, dim=0)
                    pseudo_var = torch.var(out, dim=0)
                    pseudo_mean = pseudo_mean * mask
                    pseudo_var = pseudo_var * mask

                # bs = pseudo_mean.shape[0]
                # random = np.random.randint(0, 8, size=bs)
                # transformed_in = torch.zeros_like(ul_input)
                # transformed_out = torch.zeros_like(pseudo_mean)
                # transformed_var = torch.zeros_like(pseudo_var)
                # for i in range(bs):
                #     transformed_in[i, ...] = augmentation(ul_input[i, ...], case=random[i])
                #     transformed_out[i, ...] = augmentation(pseudo_mean[i, ...], case=random[i])
                #     transformed_var[i, ...] = augmentation(pseudo_var[i, ...], case=random[i])

                ul_out = net(ul_input)
                ul_out = ul_out * mask

                var_flatten = torch.flatten(pseudo_var, 2)
                max_var = torch.max(var_flatten, dim=2)[0].reshape(-1, 3, 1, 1)
                unc_weight = 1 - (pseudo_var / max_var)
                weight_sum = torch.sum(unc_weight)
                un_loss = torch.div(torch.sum(unl_loss(ul_out, pseudo_mean) * unc_weight), weight_sum)

                un_loss.backward()
                self.optimizer.step()
                lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('train_ul_loss', un_loss, current_iter)
                if (current_iter % self.opt['print_freq']) == 0:
                    eta_str = self.time_log(current_iter)
                    print(f"Train:[epochs:{epoch}\t iter:{current_iter}]\t lr:{lr:.3e}\t eta:{eta_str}\t"
                          f"Loss_ul:{un_loss.item():.3e}")
                if (current_iter % self.opt['test_freq']) == 0 and current_iter >= self.opt['iters'] * 0.5:
                    print("-------- Testing --------")
                    net.eval()
                    with torch.no_grad():
                        loop_loss = []
                        for batch_idx, (data, targets, _) in enumerate(test_loader):
                            data, targets = data.float().to(self.device), targets.to(self.device)
                            out = net(data)
                            out = out * mask
                            loss = self.loss(out, targets)
                            loop_loss.append(loss.item())
                        loop_loss = np.array(loop_loss).mean()
                        self.writer.add_scalar('Test_S_loss', loop_loss, current_iter)
                        print(f"Test_loss:{loop_loss:.3e}")
                    net.train()
                    if loop_loss < min_loss_s:
                        min_loss_s = loop_loss
                        self.save_model(net, "S")
                        print("Saving models.")
                    print(f">>>Min_loss:{min_loss_s:.3e}")
            epoch += 1
            self.scheduler.step()
        self.start_time = time.time()

        print(f"\n-------- ReTraining Student --------")
        net_path = osp.join(model_out_path, f"{self.opt['name']}_S.pth")
        # net_path = "/mnt/zyy/reconstruction/experiments/airfoil_stp_l1e2_10132157/model/airfoil_stp_l1e2_S.pth"
        checkpoint = torch.load(net_path)
        net.load_state_dict(checkpoint['weight'])
        self.opt['optim']['lr'] *= 0.1
        self.opt['iters'] = self.opt['iters'] / 2
        self.opt['scheduler']['T_max'] = self.opt['epoch']
        self.optimizer, self.scheduler = self.init_training(net.parameters())
        current_iter, epoch = 0, 0
        net.train()
        while current_iter < self.opt['iters']:
            for batch_idx, (data, targets, mask) in enumerate(train_loader):
                current_iter += 1
                self.optimizer.zero_grad()
                data, targets = data.float().to(self.device), targets.to(self.device)
                mask = mask.to(self.device)[0, ...]

                out = net(data)
                out = out * mask
                loss = self.loss(out, targets)
                loss.backward()
                self.optimizer.step()
                lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('train_S_loss', loss, current_iter)
                if (current_iter % self.opt['print_freq']) == 0:
                    eta_str = self.time_log(current_iter)
                    print(f"Train:[epochs:{epoch}\t iter:{current_iter}]\t lr:{lr:.3e}\t eta:{eta_str}\t"
                          f"Loss:{loss.item():.3e}")
                if (current_iter % self.opt['test_freq']) == 0:
                    print("-------- Testing --------")
                    net.eval()
                    with torch.no_grad():
                        loop_loss = []
                        for batch_idx, (data, targets, _) in enumerate(test_loader):
                            data, targets = data.float().to(self.device), targets.to(self.device)
                            out = net(data)
                            out = out * mask
                            loss = self.loss(out, targets)
                            loop_loss.append(loss.item())
                        loop_loss = np.array(loop_loss).mean()
                        self.writer.add_scalar('Test_R_loss', loop_loss, current_iter)
                        print(f"Test_loss:{loop_loss:.3e}")
                    net.train()
                    if loop_loss < min_loss_s:
                        min_loss_s = loop_loss
                        self.save_model(net, "R")
                        print("Saving models.")
                    print(f">>>Min_loss:{min_loss_s:.3e}")
            epoch += 1
            self.scheduler.step()
