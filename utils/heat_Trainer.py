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

    def loop(self, train_loader, test_loader):
        min_loss = 1e6
        current_iter, epoch = 0, 0
        self.model.train()
        while current_iter < self.opt['iters']:
            for batch_idx, (data, targets) in enumerate(train_loader):
                current_iter += 1
                self.optimizer.zero_grad()
                data, targets = data.float().to(self.device), targets.to(self.device)

                out = self.model(data)
                out, targets = out * 50, targets * 50

                loss = self.loss(out, targets)
                loss.backward()
                self.optimizer.step()
                lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('Train_loss', loss.item(), current_iter)
                if (current_iter % self.opt['print_freq']) == 0:
                    eta_str = self.time_log(current_iter)
                    print(f"Train:[epochs:{epoch}\t iter:{current_iter}]\t lr:{lr:.3e}\t eta:{eta_str}\t"
                          f"Loss:{loss.item():.3e}")
                if (current_iter % self.opt['test_freq']) == 0 and current_iter >= self.opt['iters'] * 0.5:
                    print(f"-------- Testing --------")
                    self.model.eval()
                    with torch.no_grad():
                        loop_loss = []
                        for batch_idx, (data, targets) in enumerate(test_loader):
                            data, targets = data.float().to(self.device), targets.to(self.device)

                            out = self.model(data)
                            out, targets = out * 50, targets * 50
                            loss = self.loss(out, targets)
                            loop_loss.append(loss.item())
                        loop_loss = np.array(loop_loss).mean()
                        self.writer.add_scalar('Test_loss', loop_loss, current_iter)
                        print(f"Test_loss: {loop_loss:.3e}")
                    self.model.train()
                    if loop_loss < min_loss:
                        min_loss = loop_loss
                        self.save_model(self.model, "best")
                        print("Saving models.")
                    print(f">>>Min_loss:{min_loss:.3e}")
            self.scheduler.step()
            epoch += 1

    def semi_loop(self, train_loader, ul_train_loader, test_loader):
        min_loss = 1e6
        current_iter, epoch = 0, 0
        self.model.train()
        while current_iter < self.opt['iters']:
            for batch_idx, ((l_data, targets), (ul_data, _)) in enumerate(zip(train_loader, ul_train_loader)):
                current_iter += 1
                self.optimizer.zero_grad()
                l_input, ul_input = l_data.float().to(self.device), ul_data.float().to(self.device)
                targets = targets.to(self.device)

                # CPS (co-training)
                # l_out_f, l_out_s = self.model(l_input)
                # l_out_f, l_out_s, = l_out_f * 50, l_out_s * 50
                # targets = targets * 50
                # sup_loss = self.loss(l_out_f, targets) + self.loss(l_out_s, targets)
                #
                # with torch.no_grad():
                #     f_pseudo, s_pseudo = self.model(ul_input)
                #     f_pseudo, s_pseudo = f_pseudo * 50, s_pseudo * 50
                # ul_out_f, ul_out_s = self.model(ul_input)
                # ul_out_f, ul_out_s = ul_out_f * 50, ul_out_s * 50
                # un_loss = self.loss(ul_out_f, s_pseudo) + self.loss(ul_out_s, f_pseudo)
                # loss = sup_loss + un_loss

                # mean teacher
                l_out_s, l_out_t = self.model(l_input, step=1, cur_iter=current_iter)
                ul_out_s, ul_out_t = self.model(ul_input, step=2, cur_iter=current_iter)
                l_out_s, targets = l_out_s * 50, targets * 50
                ul_out_s, ul_out_t = ul_out_s * 50, ul_out_t * 50
                sup_loss = self.loss(l_out_s, targets)
                un_loss = self.loss(ul_out_s, ul_out_t)
                loss = sup_loss + un_loss

                loss.backward()
                self.optimizer.step()
                lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('train_loss', loss, current_iter)
                if (current_iter % self.opt['print_freq']) == 0:
                    eta_str = self.time_log(current_iter)
                    print_string = f"Train:[epochs:{epoch}\t iter:{current_iter}]\t lr:{lr:.3e}\t eta:{eta_str}\t"
                    print_string += f"Loss_l:{sup_loss.item():.3e}\t Loss_ul:{un_loss.item():.3e}"
                    print(print_string)
                if (current_iter % self.opt['test_freq']) == 0:
                    print("-------- Testing --------")
                    self.model.eval()
                    with torch.no_grad():
                        loop_loss = []
                        for batch_idx, (data, targets) in enumerate(test_loader):
                            data, targets = data.float().to(self.device), targets.to(self.device)

                            # CPS (co-training)
                            # out1, out2 = self.model(data)
                            # out1, out2, targets = out1 * 50, out2 * 50, targets * 50
                            # out = (out1 + out2) / 2
                            # loss = self.loss(out, targets)

                            # mean teacher
                            out = self.model(data, step=2, cur_iter=current_iter)
                            out, targets = out * 50, targets * 50
                            loss = self.loss(out, targets)

                            loop_loss.append(loss.item())
                        loop_loss = np.array(loop_loss).mean()
                        self.writer.add_scalar('Test_loss', loop_loss, current_iter)
                        print(f"Test_loss:{loop_loss:.3e}")
                    self.model.train()
                    if loop_loss < min_loss:
                        min_loss = loop_loss
                        self.save_model(self.model, "best")
                        print("Saving models.")
                    print(f">>>Min_loss:{min_loss:.3e}")
            epoch += 1
            self.scheduler.step()

    def st_loop(self, train_loader, ul_train_loader, test_loader):
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

                # y1, y2, y3 = self.model(data)
                # out = torch.stack([y1, y2, y3], dim=0)
                # out_mean = torch.mean(out, dim=0)

                out_mean = self.model(data)

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

                            # y1, y2, y3 = self.model(data)
                            # out = torch.stack([y1, y2, y3], dim=0)
                            # out_mean = torch.mean(out, dim=0)

                            out_mean = self.model(data)

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
            self.scheduler.step()
            epoch += 1
        self.start_time = time.time()

        # S model train
        print(f"\n-------- Training Student --------")
        model_out_path = Path(self.opt['save_dir'])
        net_path = osp.join(model_out_path, f"{self.opt['name']}_T.pth")
        # net_path = "/mnt/zyy/reconstruction/experiments/heat_st_l2e2/model/heat_st_l2e2_T.pth"
        checkpoint = torch.load(net_path)
        self.model.load_state_dict(checkpoint['weight'])
        self.model.eval()
        # init teacher model
        net = self.init_model('MLP', [20, 128, 1280, 4800, 40000])
        optimizer, scheduler = self.init_training(net.parameters())
        net = net.to(self.device)
        net.train()
        current_iter, epoch = 0, 0
        unl_loss = torch.nn.L1Loss(reduction="none")
        while current_iter < self.opt['iters']:
            for batch_idx, ((l_data, targets), (ul_data, ul_targets)) in enumerate(
                    zip(train_loader, ul_train_loader)):
                current_iter += 1
                optimizer.zero_grad()
                l_input, ul_input = l_data.float().to(self.device), ul_data.float().to(self.device)
                targets, ul_targets = targets.to(self.device), ul_targets.to(self.device)

                l_out = net(l_input)
                l_out, targets = l_out * 50, targets * 50
                with torch.no_grad():
                    # y1, y2, y3 = self.model(ul_input)
                    # y1, y2, y3 = y1 * 50, y2 * 50, y3 * 50
                    # ul_targets = ul_targets * 50
                    # out = torch.stack([y1, y2, y3], dim=0)
                    # pseudo_mean = torch.mean(out, dim=0)
                    # pseudo_var = torch.var(out, dim=0)

                    pseudo_mean = self.model(ul_input)
                    pseudo_mean, ul_targets = pseudo_mean * 50, ul_targets * 50
                    pseudo_loss = self.loss(pseudo_mean, ul_targets)
                ul_out = net(ul_input)
                ul_out = ul_out * 50

                # var_flatten = torch.flatten(pseudo_var, 2)
                # max_var = torch.max(var_flatten, dim=2)[0].reshape(-1, 1, 1)
                # unc_weight = 1 - (pseudo_var / max_var)
                # weight_sum = torch.sum(unc_weight)
                # un_loss = torch.div(torch.sum(unl_loss(ul_out, pseudo_mean) * unc_weight), weight_sum)

                un_loss = self.loss(ul_out, pseudo_mean)
                sup_loss = self.loss(l_out, targets)
                loss = sup_loss + un_loss

                loss.backward()
                optimizer.step()
                lr = scheduler.get_last_lr()[0]
                self.writer.add_scalar('train_S_loss', loss, current_iter)
                if (current_iter % self.opt['print_freq']) == 0:
                    eta_str = self.time_log(current_iter)
                    print_string = f"Train:[epochs:{epoch}\t iter:{current_iter}]\t lr:{lr:.3e}\t eta:{eta_str}\t"
                    print_string += f"Loss_l:{sup_loss.item():.3e}\t Loss_ul:{un_loss.item():.3e}"
                    print(print_string)
                    print(f"pseudo_loss: {pseudo_loss.item():.3e}")
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
                        self.writer.add_scalar('Test_S_loss', loop_loss, current_iter)
                        print(f"Test_loss:{loop_loss:.3e}")
                    net.train()
                    if loop_loss < min_loss_s:
                        min_loss_s = loop_loss
                        self.save_model(net, "S")
                        print("Saving models.")
                    print(f">>>Min_loss:{min_loss_s:.3e}")
            scheduler.step()
            epoch += 1

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
        # self.opt['iters'] = 40000 # epoch=160
        # self.opt['scheduler']['gamma'] = 0.99985  # 0.963
        self.opt['iters'] = self.opt['iters'] * 2
        self.opt['scheduler']['gamma'] = 0.963  # 0.963 for 160 epoch
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
        # self.opt['iters'] = 20000 # epoch = 80
        # self.opt['scheduler']['gamma'] = 0.9998  # 0.951
        self.opt['iters'] = self.opt['iters'] / 2  # epoch = 80
        self.opt['scheduler']['gamma'] = 0.951  # 0.951 for 80 epoch
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
