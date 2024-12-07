#!/usr/bin/env python
# coding: utf-8
"""
Submit experiments for QP
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
import submitit

# set parser
parser = argparse.ArgumentParser()
parser.add_argument("--samples",
                    type=int,
                    default=8000,
                    choices=[800, 8000, 80000],
                    help="problem type")
config = parser.parse_args()

import run
print("Rosenbrock")

# weight increasing
config.penalty = 100
for size in [1000, 3000, 10000]:
    # random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # init problem
    config.size = size
    config.steepness = 50            # steepness factor
    num_blocks = config.size         # number of blocks
    train_size = config.samples      # number of train
    test_size = 100                  # number of test size
    val_size = 1000                  # number of validation size
    # hyperparameters
    hsize_dict = {10:16, 30:32, 100:64, 300:128, 1000:256, 3000:512, 10000:1024}
    config.batch_size = 64                  # batch size
    config.hlayers_sol = 5                  # number of hidden layers for solution mapping
    config.hlayers_rnd = 4                  # number of hidden layers for solution mapping
    config.hsize = hsize_dict[config.size]  # width of hidden layers for solution mapping
    config.lr = 1e-3                        # learning rate
    # parameters as input data
    b_low, b_high = 1.0, 8.0
    a_low, a_high = 0.5, 4.5
    b_train = np.random.uniform(b_low, b_high, (train_size, 1)).astype(np.float32)
    b_test  = np.random.uniform(b_low, b_high, (test_size, 1)).astype(np.float32)
    b_val   = np.random.uniform(b_low, b_high, (val_size, 1)).astype(np.float32)
    a_train = np.random.uniform(a_low, a_high, (train_size, num_blocks)).astype(np.float32)
    a_test  = np.random.uniform(a_low, a_high, (test_size, num_blocks)).astype(np.float32)
    a_val   = np.random.uniform(a_low, a_high, (val_size, num_blocks)).astype(np.float32)
    # nm datasets
    from neuromancer.dataset import DictDataset
    data_train = DictDataset({"b":b_train, "a":a_train}, name="train")
    data_test = DictDataset({"b":b_test, "a":a_test}, name="test")
    data_val = DictDataset({"b":b_val, "a":a_val}, name="dev")
    # torch dataloaders
    from torch.utils.data import DataLoader
    batch_size = 64
    loader_train = DataLoader(data_train, batch_size, num_workers=0, collate_fn=data_train.collate_fn, shuffle=True)
    loader_test = DataLoader(data_test, batch_size, num_workers=0, collate_fn=data_test.collate_fn, shuffle=False)
    loader_val = DataLoader(data_val, batch_size, num_workers=0, collate_fn=data_val.collate_fn, shuffle=True)
    # submit experiments
    #run.rosenbrock.rndCls(loader_train, loader_test, loader_val, config, True)
    #run.rosenbrock.rndThd(loader_train, loader_test, loader_val, config, True)
    #run.rosenbrock.rndSte(loader_train, loader_test, loader_val, config, True)
    print(config)
    # rounding classification
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(slurm_additional_parameters={"account": "def-khalile2",
                                                            "constraint": "v100l"},
                               timeout_min=180,
                               mem_gb=64,
                               cpus_per_task=16,
                               gpus_per_node=1)
    job = executor.submit(run.rosenbrock.rndCls, loader_train, loader_test, loader_val, config, True)
    print(f"Submitted job with ID: {job.job_id}")
    # learnable threshold
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(slurm_additional_parameters={"account": "def-khalile2",
                                                            "constraint": "v100l"},
                               timeout_min=180,
                               mem_gb=64,
                               cpus_per_task=16,
                               gpus_per_node=1)
    job = executor.submit(run.rosenbrock.rndThd, loader_train, loader_test, loader_val, config, True)
    print(f"Submitted job with ID: {job.job_id}")
    # STE Rounding
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(slurm_additional_parameters={"account": "def-khalile2",
                                                            "constraint": "v100l"},
                               timeout_min=180,
                               mem_gb=64,
                               cpus_per_task=16,
                               gpus_per_node=1)
    job = executor.submit(run.rosenbrock.rndSte, loader_train, loader_test, loader_val, config, True)
    print(f"Submitted job with ID: {job.job_id}")

for penalty in [1, 5, 10, 50, 500, 1000]:
    config.penalty = penalty
    for size in [1000, 3000, 10000]:
        # random seed
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # init problem
        config.size = size
        config.steepness = 50            # steepness factor
        num_blocks = config.size         # number of blocks
        train_size = config.samples      # number of train
        test_size = 100                  # number of test size
        val_size = 1000                  # number of validation size
        # hyperparameters
        hsize_dict = {10:16, 30:32, 100:64, 300:128, 1000:256, 3000:512, 10000:1024}
        config.batch_size = 64                  # batch size
        config.hlayers_sol = 5                  # number of hidden layers for solution mapping
        config.hlayers_rnd = 4                  # number of hidden layers for solution mapping
        config.hsize = hsize_dict[config.size]  # width of hidden layers for solution mapping
        config.lr = 1e-3                        # learning rate
        # parameters as input data
        b_low, b_high = 1.0, 8.0
        a_low, a_high = 0.5, 4.5
        b_train = np.random.uniform(b_low, b_high, (train_size, 1)).astype(np.float32)
        b_test  = np.random.uniform(b_low, b_high, (test_size, 1)).astype(np.float32)
        b_val   = np.random.uniform(b_low, b_high, (val_size, 1)).astype(np.float32)
        a_train = np.random.uniform(a_low, a_high, (train_size, num_blocks)).astype(np.float32)
        a_test  = np.random.uniform(a_low, a_high, (test_size, num_blocks)).astype(np.float32)
        a_val   = np.random.uniform(a_low, a_high, (val_size, num_blocks)).astype(np.float32)
        # nm datasets
        from neuromancer.dataset import DictDataset
        data_train = DictDataset({"b":b_train, "a":a_train}, name="train")
        data_test = DictDataset({"b":b_test, "a":a_test}, name="test")
        data_val = DictDataset({"b":b_val, "a":a_val}, name="dev")
        # torch dataloaders
        from torch.utils.data import DataLoader
        batch_size = 64
        loader_train = DataLoader(data_train, batch_size, num_workers=0, collate_fn=data_train.collate_fn, shuffle=True)
        loader_test = DataLoader(data_test, batch_size, num_workers=0, collate_fn=data_test.collate_fn, shuffle=False)
        loader_val = DataLoader(data_val, batch_size, num_workers=0, collate_fn=data_val.collate_fn, shuffle=True)
        # submit experiments
        #run.rosenbrock.rndCls(loader_train, loader_test, loader_val, config)
        #run.rosenbrock.rndThd(loader_train, loader_test, loader_val, config)
        #run.rosenbrock.rndSte(loader_train, loader_test, loader_val, config)
        print(config)
        # rounding classification
        executor = submitit.AutoExecutor(folder="logs")
        executor.update_parameters(slurm_additional_parameters={"account": "def-khalile2",
                                                                "constraint": "v100l"},
                                   timeout_min=180,
                                   mem_gb=64,
                                   cpus_per_task=16,
                                   gpus_per_node=1)
        job = executor.submit(run.rosenbrock.rndCls, loader_train, loader_test, loader_val, config)
        print(f"Submitted job with ID: {job.job_id}")
        # learnable threshold
        executor = submitit.AutoExecutor(folder="logs")
        executor.update_parameters(slurm_additional_parameters={"account": "def-khalile2",
                                                                "constraint": "v100l"},
                                   timeout_min=180,
                                   mem_gb=64,
                                   cpus_per_task=16,
                                   gpus_per_node=1)
        job = executor.submit(run.rosenbrock.rndThd, loader_train, loader_test, loader_val, config)
        print(f"Submitted job with ID: {job.job_id}")
        # STE Rounding
        executor = submitit.AutoExecutor(folder="logs")
        executor.update_parameters(slurm_additional_parameters={"account": "def-khalile2",
                                                                "constraint": "v100l"},
                                   timeout_min=180,
                                   mem_gb=64,
                                   cpus_per_task=16,
                                   gpus_per_node=1)
        job = executor.submit(run.rosenbrock.rndSte, loader_train, loader_test, loader_val, config)
        print(f"Submitted job with ID: {job.job_id}")
