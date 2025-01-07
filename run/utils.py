#!/usr/bin/env python
# coding: utf-8
"""
Utlities
"""

import torch
from src.problem.neuromancer.trainer import trainer

def train(components, loss_fn, loader_train, loader_val, lr):
    epochs = 200                    # number of training epochs
    patience = 20                   # number of epochs with no improvement in eval metric to allow before early stopping
    warmup = 40                 # number of epochs to wait before enacting early stopping policies
    optimizer = torch.optim.AdamW(components.parameters(), lr=lr)
    # create a trainer for the problem
    my_trainer = trainer(components, loss_fn, optimizer, epochs=epochs,
                         patience=patience, warmup=warmup, device="cuda")
    # training for the rounding problem
    my_trainer.train(loader_train, loader_val)
