#!coding:utf-8
import os.path as osp
import time
from pathlib import Path

import numpy as np
import torch

from utils.Trainer import Trainer


class SubTrainer(Trainer):
    def __init__(self, opt, writer):
        super(SubTrainer, self).__init__(opt, writer)
        # self.model = self.init_model('MT_MLP', [20, 128, 1280, 4800, 40000])
        self.model = self.init_model('MLP', [20, 128, 1280, 4800, 40000])
        self.optimizer, self.scheduler = self.init_training(self.model.parameters())
        self.model = self.model.to(self.device)

    def stp_loop(self, train_loader, ul_train_loader, test_loader):
        min_loss_t, min_loss_s = 1e6, 1e6
        current_iter, epoch = 0, 0
        # T model train
        self.model.train()
        print(f"-------- Training Teacher --------")
        while current_iter < self.opt['iters']:
            for batch_idx, (data, targets) in enumerate(train_loader):
                current_iter += 1
                self.optimizer.zero_grad()
                data, targets = data.float().to(self.device), targets.to(self.device)

                y1, y2, y3 = self.model(data)
                out = torch.stack([y1, y2, y3], dim=0)
                out_mean = torch.mean(out, dim=0)
                out_mean, targets = out_mean * 50, targets * 50

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
                        for batch_idx, (data, targets) in enumerate(test_loader):
                            data, targets = data.float().to(self.device), targets.to(self.device)

                            y1, y2, y3 = self.model(data)
                            out = torch.stack([y1, y2, y3], dim=0)
                            out_mean = torch.mean(out, dim=0)
                            out_mean, targets = out_mean * 50, targets * 50

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
        # net_path = "/mnt/zyy/reconstruction/experiments/heat_stp_l1e1_teacher_e3/model/heat_stp_l1e1_teacher_e3_T.pth"
        checkpoint = torch.load(net_path)
        self.model.load_state_dict(checkpoint['weight'])
        self.model.eval()
        # init student model
        self.opt['iters'] = self.opt['iters'] * 2
        self.opt['scheduler']['gamma'] = 0.963  
        net = self.init_model('MLP', [20, 128, 1280, 4800, 40000])
        net = net.to(self.device)
        self.optimizer, self.scheduler = self.init_training(net.parameters())
        net.train()
        current_iter, epoch = 0, 0
        unl_loss = torch.nn.L1Loss(reduction="none")
        while current_iter < self.opt['iters']:
            for batch_idx, (ul_data, _) in enumerate(ul_train_loader):
                current_iter += 1
                self.optimizer.zero_grad()
                ul_input = ul_data.float().to(self.device)

                with torch.no_grad():
                    y1, y2, y3 = self.model(ul_input)
                    y1, y2, y3 = y1 * 50, y2 * 50, y3 * 50
                    out = torch.stack([y1, y2, y3], dim=0)
                    pseudo_mean = torch.mean(out, dim=0)
                    pseudo_var = torch.var(out, dim=0)

                ul_out = net(ul_input)
                ul_out = ul_out * 50

                var_flatten = torch.flatten(pseudo_var, 2)
                max_var = torch.max(var_flatten, dim=2)[0].reshape(-1, 1, 1)
                unc_weight = 1 - (pseudo_var / max_var)
                weight_sum = torch.sum(unc_weight)
                un_loss = torch.div(torch.sum(unl_loss(ul_out, pseudo_mean) * unc_weight), weight_sum)
                # un_loss = self.loss(ul_out, pseudo_mean)

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
                        for batch_idx, (data, targets) in enumerate(test_loader):
                            data, targets = data.float().to(self.device), targets.to(self.device)
                            out = net(data)
                            out, targets = out * 50, targets * 50

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
        # net_path = "/mnt/zyy/reconstruction/experiments/heat_stp_l2e2/model/heat_stp_l2e2_S.pth"
        checkpoint = torch.load(net_path)
        net.load_state_dict(checkpoint['weight'])
        self.opt['optim']['lr'] *= 0.01
        self.opt['iters'] = self.opt['iters'] / 2  
        self.opt['scheduler']['gamma'] = 0.951  
        self.optimizer, self.scheduler = self.init_training(net.parameters())
        current_iter, epoch = 0, 0
        net.train()
        while current_iter < self.opt['iters']:
            for batch_idx, (data, targets) in enumerate(train_loader):
                current_iter += 1
                if current_iter > self.opt['iters']:
                    break
                self.optimizer.zero_grad()
                data, targets = data.float().to(self.device), targets.to(self.device)

                out = net(data)
                out, targets = out * 50, targets * 50
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
                        for batch_idx, (data, targets) in enumerate(test_loader):
                            data, targets = data.float().to(self.device), targets.to(self.device)
                            out = net(data)
                            out, targets = out * 50, targets * 50
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
