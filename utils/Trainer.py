#!coding:utf-8
import datetime
import time
from pathlib import Path

import torch

import architectures
from utils.util import create_optim, create_lr_scheduler, create_loss_fn


class Trainer:
    def __init__(self, opt, writer):
        self.opt = opt
        self.device = "cuda"
        self.loss = create_loss_fn(self.opt['loss'])
        self.writer = writer
        self.start_time = time.time()
        self.start_iter = 0

    @staticmethod
    def init_model(model_name, *kw):
        model = getattr(architectures, model_name)
        model = model(*kw)
        return model

    def init_training(self, model_params):
        optimizer = create_optim(model_params, self.opt['optim'])
        scheduler = create_lr_scheduler(optimizer, self.opt['scheduler'])
        return optimizer, scheduler

    def save_model(self, model, iters):
        if 'save_dir' in self.opt:
            model_out_path = Path(self.opt['save_dir'])
            state = {"iters": iters,
                     "weight": model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            torch.save(state, model_out_path / f"{self.opt['name']}_{iters}.pth")
        else:
            raise Exception("save_dir is none!")

    def time_log(self, current_iter):
        total_time = time.time() - self.start_time
        time_sec_avg = total_time / (current_iter - self.start_iter)
        eta_sec = time_sec_avg * (self.opt['iters'] - current_iter)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        return eta_str

    def loop(self, train_loader, test_loader):
        pass

    def semi_loop(self, train_loader, ul_train_loader, test_loader):
        pass

    def st_loop(self, train_loader, ul_train_loader, test_loader):
        pass

    def st_loop1(self, train_loader, ul_train_loader, test_loader):
        pass
