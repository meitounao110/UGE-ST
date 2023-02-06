#!coding:utf-8
import sys
from collections import OrderedDict

import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim import lr_scheduler


def parse(opt_path):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)
    return opt


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def create_loss_fn(loss_type):
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'mae':
        criterion = nn.L1Loss()
    else:
        raise ValueError(f'Wrong loss type.')
    return criterion


def create_optim(params, opt):
    if opt['type'] == 'sgd':
        optimizer = optim.SGD(params, opt['lr'],
                              momentum=opt['momentum'],
                              weight_decay=opt['weight_decay'])
    elif opt['type'] == 'adam':
        optimizer = optim.AdamW(params, opt['lr'])
    else:
        raise ValueError(f'Wrong optim type.')
    return optimizer


def create_lr_scheduler(optimizer, opt):
    if opt['type'] == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt['T_max'],
                                                   eta_min=opt['min_lr'])
    elif opt['type'] == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=opt['gamma'])
    elif opt['type'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt['gamma'], gamma=opt['gamma'])
    else:
        raise ValueError(f'Wrong scheduler type.')
    return scheduler
