import argparse
import datetime
import math
import os
import random
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import *


def parse_dict_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--num_gpu', type=str, default=0)
    args = parser.parse_args()
    print("Using these args:", args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.num_gpu
    opt = parse(args.opt)
    opt["yaml_path"] = args.opt
    return opt


def dataloader_sup(datasets, datadir, opt):
    dataset = eval(datasets['name'] + "_datasets." + datasets['type'])
    eval_set = dataset(root=datadir, split="test", opt=opt)
    eval_loader = DataLoader(eval_set,
                             batch_size=opt['val_batch_size'],
                             shuffle=False,
                             num_workers=opt['workers'],
                             pin_memory=False,
                             drop_last=False)
    train_set = dataset(root=datadir, split="train", opt=opt)
    train_sampler = EnlargedSampler(train_set, 1, 0, opt['dataset_enlarge_ratio'])
    train_loader = DataLoader(train_set,
                              sampler=train_sampler,
                              batch_size=opt['batch_size'],
                              num_workers=opt['workers'],
                              drop_last=True,
                              pin_memory=False)
    iters_epochs = len(train_loader)
    train_num = len(train_set)
    test_num = len(eval_set)
    opt["iters"] = opt["epoch"] * iters_epochs
    opt['print_freq'] = opt['print_freq'] * iters_epochs
    opt['test_freq'] = opt['test_freq'] * iters_epochs
    print(f"Training statistics:\n "
          f"\tNumber of test data:{test_num}\n"
          f"\tNumber of train data:{train_num}\n"
          f"\tDataset_enlarge_ratio:{opt['dataset_enlarge_ratio']}\n"
          f"\tRequire labeled iter number per epoch:{iters_epochs}\n")

    return train_loader, eval_loader


def dataloader_semi(datasets, datadir, opt):
    dataset = eval(datasets['name'] + "_datasets." + datasets['type'])
    eval_set = dataset(root=datadir, split="test", opt=opt)
    eval_loader = DataLoader(eval_set,
                             batch_size=opt['val_batch_size'],
                             shuffle=False,
                             num_workers=opt['workers'],
                             pin_memory=False,
                             drop_last=False)
    ul_train_set = dataset(root=datadir, split="ul_train", opt=opt)
    ul_train_loader = DataLoader(ul_train_set,
                                 batch_size=opt['ul_batch_size'],
                                 shuffle=True,
                                 num_workers=opt['workers'],
                                 drop_last=True,
                                 pin_memory=False)
    train_set = dataset(root=datadir, split="train", opt=opt)
    opt['dataset_enlarge_ratio'] = math.ceil(len(ul_train_loader) / (len(train_set) / opt['batch_size']))
    train_sampler = EnlargedSampler(train_set, 1, 0, opt['dataset_enlarge_ratio'])
    train_loader = DataLoader(train_set,
                              sampler=train_sampler,
                              batch_size=opt['batch_size'],
                              num_workers=opt['workers'],
                              drop_last=True,
                              pin_memory=False)
    iters_epochs = len(train_loader)
    ul_iters_epochs = len(ul_train_loader)
    train_num = len(train_set)
    test_num = len(eval_set)
    ul_train_num = len(ul_train_set)
    opt["iters"] = opt["epoch"] * ul_iters_epochs
    opt['print_freq'] = opt['print_freq'] * ul_iters_epochs
    opt['test_freq'] = opt['test_freq'] * ul_iters_epochs
    print(f"Training statistics:\n "
          f"\tNumber of test data:{test_num}\n"
          f"\tNumber of train data:{train_num}\n"
          f"\tNumber of unlabeled train data:{ul_train_num}\n"
          f"\tDataset_enlarge_ratio:{opt['dataset_enlarge_ratio']}\n"
          f"\tRequire labeled iter number per epoch:{iters_epochs}\n"
          f"\tRequire unlabeled iter number per epoch:{ul_iters_epochs}\n")

    return train_loader, eval_loader, ul_train_loader


def main(opt):
    now = datetime.datetime.now().strftime('%m%d%H%M')
    writer = SummaryWriter(f"./logs/{opt['name']}")
    log_dir = f"./experiments/{opt['name']}"
    if os.path.exists(log_dir):
        log_dir += f'_{now}'
    os.mkdir(log_dir)
    shutil.copyfile(opt["yaml_path"], os.path.join(log_dir, "options.yml"))
    shutil.copyfile("/mnt/zyy/reconstruction/utils/" + opt['dataset']['name'] + "_Trainer.py",
                    os.path.join(log_dir, opt['dataset']['name'] + "_Trainer.py"))
    sys.stdout = Logger(os.path.join(log_dir, "log.txt"), stream=sys.stdout)
    sys.stderr = Logger(os.path.join(log_dir, "log.txt"), stream=sys.stderr)
    opt["save_dir"] = os.path.join(log_dir, "model")

    seed = 114514
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = True
    train = eval(opt['dataset']['name'] + "_Trainer")
    trainer = train.SubTrainer(opt, writer)
    if 'sup' in opt['name']:
        train_loader, eval_loader = dataloader_sup(opt['dataset'], opt['data_dir'], opt=opt)
        trainer.loop(train_loader, eval_loader)
        # trainer.diff_loop(train_loader, eval_loader)
        # trainer.twin_loop(train_loader, eval_loader)
        # trainer.pod_loop(train_loader, eval_loader)
    else:
        train_loader, eval_loader, ul_train_loader = dataloader_semi(opt['dataset'], opt['data_dir'], opt=opt)
        # trainer.semi_loop(train_loader, ul_train_loader, eval_loader)
        trainer.st_loop(train_loader, ul_train_loader, eval_loader)
        # trainer.stp_loop(train_loader, ul_train_loader, eval_loader)


if __name__ == "__main__":
    opt = parse_dict_args()
    main(opt)
