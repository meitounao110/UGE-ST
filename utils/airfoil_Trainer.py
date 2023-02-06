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


def augmentation(img, case):
    if case == 0:
        img_a = img
    elif case == 1:
        img_a = TF.rotate(img, angle=90)
    elif case == 2:
        img_a = TF.rotate(img, angle=180)
    elif case == 3:
        img_a = TF.rotate(img, angle=270)
    elif case == 4:
        img_a = TF.vflip(img)  # 上下翻转
    elif case == 5:
        img_a = TF.hflip(img)  # 水平翻转
    elif case == 6:
        img_a = TF.rotate(TF.vflip(img), angle=90)
    elif case == 7:
        img_a = TF.rotate(TF.hflip(img), angle=90)
    else:
        raise ValueError("no augmentation")
    return img_a


def normal_distribution(x, mean=0, sigma=1):
    return torch.div(torch.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))), (math.sqrt(2 * torch.pi) * sigma))


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

    def loop(self, train_loader, test_loader):
        min_loss = 1e6
        current_iter, epoch = 0, 0
        self.model.train()
        criterion = QuantileRegressionLoss()
        while current_iter < self.opt['iters']:
            for batch_idx, (data, targets, mask) in enumerate(train_loader):
                current_iter += 1
                self.optimizer.zero_grad()
                data, targets = data.float().to(self.device), targets.to(self.device)
                mask = mask.to(self.device)[0, ...]
                ####### 分位数
                # taus = torch.rand(data.size(0), 1).float().to(self.device)
                # taus = repeat(taus, 'b c -> b c h w', h=data.size(2), w=data.size(3))
                # inputs_taus = torch.cat((data, taus), 1)
                # out = self.model(inputs_taus)
                # out = out * mask
                # loss = criterion(targets, out, taus)

                out = self.model(data)
                ####### 负对数似然
                mean_out = out[:, 0:3, ...]
                mean_out = mean_out * mask
                softplus = torch.nn.Softplus()
                var_out = softplus(out[:, 3:, ...]) + 1e-8
                # w_epsilon = Normal(0, 1).sample(mean_out.shape).to(self.device)
                w_epsilon = Normal(mean_out, var_out)
                # x_2 = (targets - mean_out) / var_out
                loss = torch.mean(- w_epsilon.log_prob(targets))
                ###############

                loss1 = self.loss(mean_out, targets)

                loss.backward()
                self.optimizer.step()
                lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('Train_loss', loss.item(), current_iter)
                if (current_iter % self.opt['print_freq']) == 0:
                    eta_str = self.time_log(current_iter)
                    print(f"Train:[epochs:{epoch}\t iter:{current_iter}]\t lr:{lr:.3e}\t eta:{eta_str}\t"
                          f"Loss:{loss.item():.3e}\t MAE:{loss1.item()}")
                if (current_iter % self.opt['test_freq']) == 0 and current_iter >= self.opt['iters'] * 0.5:
                    print(f"-------- Testing --------")
                    self.model.eval()
                    with torch.no_grad():
                        loop_loss = []
                        for batch_idx, (data, targets, _) in enumerate(test_loader):
                            data, targets = data.float().to(self.device), targets.to(self.device)

                            ### 分位数
                            # taus = torch.zeros(data.size(0), 1).fill_(0.5).float().to(self.device)
                            # taus = repeat(taus, 'b c -> b c h w', h=data.size(2), w=data.size(3))
                            # inputs_taus = torch.cat((data, taus), 1)
                            # out = self.model(inputs_taus)

                            out = self.model(data)
                            out = out[:, 0:3, ...]

                            out = out * mask
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
            epoch += 1
            self.scheduler.step()

    def semi_loop(self, train_loader, ul_train_loader, test_loader):
        min_loss = 1e6
        current_iter, epoch = 0, 0
        self.model.train()
        while current_iter < self.opt['iters']:
            for batch_idx, ((l_data, targets, mask), (ul_data, _, _)) in enumerate(
                    zip(train_loader, ul_train_loader)):
                current_iter += 1
                self.optimizer.zero_grad()
                l_input, ul_input = l_data.float().to(self.device), ul_data.float().to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)[0, ...]

                # CPS (co-training)
                # l_out_f, l_out_s = self.model(l_input)
                # l_out_f, l_out_s = l_out_f * mask, l_out_s * mask
                # sup_loss1 = self.loss(l_out_f, targets)
                # sup_loss2 = self.loss(l_out_s, targets)
                #
                # with torch.no_grad():
                #     f_pseudo, s_pseudo = self.model(ul_input)
                #     f_pseudo, s_pseudo = f_pseudo * mask, s_pseudo * mask
                # ul_out_f, ul_out_s = self.model(ul_input)
                # ul_out_f, ul_out_s = ul_out_f * mask, ul_out_s * mask
                # un_loss1 = self.loss(ul_out_f, s_pseudo)
                # un_loss2 = self.loss(ul_out_s, f_pseudo)
                # loss = sup_loss1 + sup_loss2 + (un_loss1 + un_loss2)

                # mean teacher
                l_out_s, l_out_t = self.model(l_input, step=1, cur_iter=current_iter)
                ul_out_s, ul_out_t = self.model(ul_input, step=2, cur_iter=current_iter)
                l_out_s = l_out_s * mask
                ul_out_s, ul_out_t = ul_out_s * mask, ul_out_t * mask
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
                        for batch_idx, (data, targets, _) in enumerate(test_loader):
                            data, targets = data.float().to(self.device), targets.to(self.device)

                            # CPS (co-training)
                            # out1, out2 = self.model(data)
                            # out1, out2 = out1 * mask, out2 * mask
                            # out = (out1 + out2) / 2
                            # loss = self.loss(out, targets)

                            # mean teacher
                            out = self.model(data, step=2, cur_iter=current_iter)
                            out = out * mask
                            loss = self.loss(out, targets)

                            loop_loss.append(loss.item())
                        loop_loss = np.array(loop_loss).mean()
                        self.writer.add_scalar('Test_loss', loop_loss, current_iter)
                        print_string = f"Test_loss:{loop_loss:.3e}"
                        print(print_string)
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
            for batch_idx, (data, targets, mask) in enumerate(train_loader):
                current_iter += 1
                self.optimizer.zero_grad()
                data, targets = data.float().to(self.device), targets.to(self.device)
                mask = mask.to(self.device)[0, ...]

                # y1, y2, y3 = self.model(data)
                # out = torch.stack([y1, y2, y3], dim=0)
                # out_mean = torch.mean(out, dim=0)

                out_mean = self.model(data, step=1)
                out_mean = out_mean * mask

                out_detach = out_mean.detach()
                data_out = torch.cat((data, out_detach), dim=1)

                out_err = self.model(data_out.float().to(self.device), step=2)
                out_err = out_err * mask
                err = targets - out_detach
                loss_err = self.loss(out_err, err)
                loss_sup = self.loss(out_mean, targets)
                loss = loss_sup + loss_err

                loss.backward()
                self.optimizer.step()
                lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('train_T_loss', loss, current_iter)
                if (current_iter % self.opt['print_freq']) == 0:
                    eta_str = self.time_log(current_iter)
                    print_string = f"Train:[epochs:{epoch}\t iter:{current_iter}]\t lr:{lr:.3e}\t eta:{eta_str}\t"
                    print_string += f"Loss:{loss.item():.3e}\t loss_sup:{loss_sup.item():.3e}\t loss_err:{loss_err.item():.3e}"
                    print(print_string)
                if (current_iter % self.opt['test_freq']) == 0 and current_iter >= self.opt['iters'] * 0.1:
                    print(f"-------- Testing --------")
                    self.model.eval()
                    with torch.no_grad():
                        loop_loss = []
                        for batch_idx, (data, targets, _) in enumerate(test_loader):
                            data, targets = data.float().to(self.device), targets.to(self.device)

                            # y1, y2, y3 = self.model(data)
                            # out = torch.stack([y1, y2, y3], dim=0)
                            # out_mean = torch.mean(out, dim=0)

                            out_mean = self.model(data, step=1)
                            out_mean = out_mean * mask

                            out_detach = out_mean.detach()
                            data_out = torch.cat((data, out_detach), dim=1)
                            out_err = self.model(data_out.float().to(self.device), step=2)
                            out_err = out_err * mask

                            out_mean = out_mean + out_err

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
        # net_path = "/mnt/zyy/reconstruction/experiments/airfoil_STP/airfoil_st_l2e2/model/airfoil_st_l2e2_T.pth"
        checkpoint = torch.load(net_path)
        self.model.load_state_dict(checkpoint['weight'])
        self.model.eval()
        # init student model
        net = self.init_model('UNet', 3, 3)
        net = net.to(self.device)
        optimizer, scheduler = self.init_training(net.parameters())
        net.train()
        current_iter, epoch = 0, 0
        unl_loss = torch.nn.L1Loss(reduction="none")
        while current_iter < self.opt['iters']:
            for batch_idx, ((l_data, targets, mask), (ul_data, ul_targets, _)) in enumerate(
                    zip(train_loader, ul_train_loader)):
                current_iter += 1
                optimizer.zero_grad()
                l_input, ul_input = l_data.float().to(self.device), ul_data.float().to(self.device)
                targets = targets.to(self.device)
                ul_targets = ul_targets.to(self.device)
                mask = mask.to(self.device)[0, ...]

                l_out = net(l_input)
                l_out = l_out * mask
                with torch.no_grad():
                    # y1, y2, y3 = self.model(ul_input)
                    # out = torch.stack([y1, y2, y3], dim=0)
                    # pseudo_mean = torch.mean(out, dim=0)
                    # pseudo_var = torch.var(out, dim=0)

                    pseudo_mean = self.model(ul_input, step=1)
                    pseudo_mean = pseudo_mean * mask

                    data_out = torch.cat((ul_input, pseudo_mean), dim=1)
                    out_err = self.model(data_out.float().to(self.device), step=2)
                    out_err = out_err * mask

                    pseudo_mean = pseudo_mean + out_err

                    # pseudo_var = pseudo_var * mask
                    pseudo_loss = self.loss(pseudo_mean, ul_targets)

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

                # var_flatten = torch.flatten(pseudo_var, 2)
                # # max_var = torch.max(pseudo_var, dim=2)[0].reshape(-1, 3, 1)
                # max_var = torch.max(var_flatten, dim=2)[0].reshape(-1, 3, 1, 1)
                # unc_weight = 1 - (pseudo_var / max_var)
                # weight_sum = torch.sum(unc_weight)
                # un_loss = torch.div(torch.sum(unl_loss(ul_out, pseudo_mean) * unc_weight), weight_sum)

                out_err = torch.abs(out_err)
                var_flatten = torch.flatten(out_err, 2)
                max_var = torch.max(var_flatten, dim=2)[0].reshape(-1, 3, 1, 1)
                unc_weight = 1 - (out_err / max_var)
                weight_sum = torch.sum(unc_weight)
                un_loss = torch.div(torch.sum(unl_loss(ul_out, pseudo_mean) * unc_weight), weight_sum)

                # un_loss = self.loss(ul_out, pseudo_mean)

                sup_loss = self.loss(l_out, targets)
                loss = sup_loss + 1 * un_loss

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
                        for batch_idx, (data, targets, _) in enumerate(test_loader):
                            data, targets = data.float().to(self.device), targets.to(self.device)
                            out = net(data)
                            out = out * mask

                            out = out.detach()
                            data_out = torch.cat((data, out), dim=1)
                            out_err = self.model(data_out.float().to(self.device), step=2)
                            out_err = out_err * mask

                            out = out + out_err

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
            scheduler.step()

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
