#!/usr/bin/env python
# coding: utf-8
"""
Submit experiments for INC
"""
import argparse
import itertools
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

# random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set parser
parser = argparse.ArgumentParser()
parser.add_argument("--size",
                    type=int,
                    default=5,
                    choices=[5, 10, 20, 50, 100, 200, 500, 1000],
                    help="problem size")
parser.add_argument("--penalty",
                    type=float,
                    default=100,
                    help="penalty weight")
parser.add_argument("--project",
                    action="store_true",
                    help="project gradient")
parser.add_argument("--warmstart",
                    action="store_true",
                    help="warm start")
config = parser.parse_args()

# init problem
num_var = config.size            # number of variables
num_ineq = config.size           # number of constraints
train_size = 8000                # number of train
test_size = 1000                 # number of test size
val_size = 1000                  # number of validation size
num_data = train_size + test_size + val_size

# hyperparameters
hsize_dict = {5:16, 10:32, 20:64, 50:128, 100:256, 200:512, 500:1024, 1000:2048}
config.batch_size = 64                  # batch size
config.hlayers_sol = 5                  # number of hidden layers for solution mapping
config.hlayers_rnd = 4                  # number of hidden layers for solution mapping
config.hsize = hsize_dict[config.size]  # width of hidden layers for solution mapping
config.lr = 1e-3                        # learning rate

# data sample from uniform distribution
b_samples = torch.from_numpy(np.random.uniform(-1, 1, size=(num_data, num_ineq))).float()
d_samples = torch.from_numpy(np.random.uniform(-0.1, 0.1, size=(num_data, num_ineq))).float()
data = {"b":b_samples, "d":d_samples}
# data split
from src.utlis import data_split
data_train, data_test, data_val = data_split(data, test_size=test_size, val_size=val_size)
# torch dataloaders
from torch.utils.data import DataLoader
loader_train = DataLoader(data_train, config.batch_size, num_workers=0,
                          collate_fn=data_train.collate_fn, shuffle=True)
loader_test  = DataLoader(data_test, config.batch_size, num_workers=0,
                          collate_fn=data_test.collate_fn, shuffle=False)
loader_val   = DataLoader(data_val, config.batch_size, num_workers=0,
                          collate_fn=data_val.collate_fn, shuffle=False)

import run
print("Simple Non-Convex")
if config.project is True and config.warmstart is True:
    print("Warm Starting:")
    run.nonconvex.exact(loader_test, config)
    run.nonconvex.rndCls(loader_train, loader_test, loader_val, config)
    run.nonconvex.rndThd(loader_train, loader_test, loader_val, config)
elif config.project is True:
    print("Feasibility projection:")
    run.nonconvex.rndCls(loader_train, loader_test, loader_val, config)
    run.nonconvex.rndThd(loader_train, loader_test, loader_val, config)
    run.nonconvex.lrnRnd(loader_train, loader_test, loader_val, config)
    run.nonconvex.rndSte(loader_train, loader_test, loader_val, config)
else:
    run.nonconvex.exact(loader_test, config)
    run.nonconvex.relRnd(loader_test, config)
    run.nonconvex.root(loader_test, config)
    run.nonconvex.rndCls(loader_train, loader_test, loader_val, config)
    run.nonconvex.rndThd(loader_train, loader_test, loader_val, config)
    run.nonconvex.lrnRnd(loader_train, loader_test, loader_val, config)
    run.nonconvex.rndSte(loader_train, loader_test, loader_val, config)
