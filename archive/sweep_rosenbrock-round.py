import time
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import neuromancer as nm
from neuromancer.dataset import DictDataset

from problem.neural import probRosenbrock
from problem.solver import exactRosenbrock
from model.layer import netFC
from model.round import roundModel

# set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# turn off warning
import logging
logging.getLogger("pyomo.core").setLevel(logging.ERROR)

# dataset
num_data = 5000   # number of data
num_vars = 5      # number of decision variables
num_ints = 5      # number of integer decision variables
test_size = 1000  # number of test size
val_size = 1000   # number of validation size
train_size = num_data - test_size - val_size
# parameters as input data
p_train = np.random.uniform(1.0, 6.0, (train_size, 1)).astype(np.float32)
a_train = np.random.uniform(0.2, 1.2, (train_size, num_vars-1)).astype(np.float32)
p_test = np.random.uniform(1.0, 6.0, (test_size, 1)).astype(np.float32)
a_test = np.random.uniform(0.2, 1.2, (test_size, num_vars-1)).astype(np.float32)
p_dev = np.random.uniform(1.0, 6.0, (val_size, 1)).astype(np.float32)
a_dev = np.random.uniform(0.2, 1.2, (val_size, num_vars-1)).astype(np.float32)
# nm datasets
data_train = DictDataset({"p":p_train, "a":a_train}, name="train")
data_test = DictDataset({"p":p_test, "a":a_test}, name="test")
data_dev = DictDataset({"p":p_dev, "a":a_dev}, name="dev")
# torch dataloaders
loader_train = DataLoader(data_train, batch_size=32, num_workers=0, collate_fn=data_train.collate_fn, shuffle=True)
loader_test = DataLoader(data_test, batch_size=32, num_workers=0, collate_fn=data_test.collate_fn, shuffle=False)
loader_dev = DataLoader(data_dev, batch_size=32, num_workers=0, collate_fn=data_dev.collate_fn, shuffle=True)

# exact solver
model = exactRosenbrock(n_vars=num_vars, n_integers=num_ints)

def getNMProb(round_module, hidden_dims):
    # parameters
    p = nm.constraint.variable("p")
    a = nm.constraint.variable("a")
    # variables
    x_bar = nm.constraint.variable("x_bar")
    x_rnd = nm.constraint.variable("x_rnd")
    # model
    obj_bar, constrs_bar = probRosenbrock(x_bar, p, a, num_vars=num_vars, alpha=100)
    obj_rnd, constrs_rnd = probRosenbrock(x_rnd, p, a, num_vars=num_vars, alpha=100)
    # define neural architecture for the solution mapping
    func = nm.modules.blocks.MLP(insize=num_vars, outsize=num_vars, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=hidden_dims)
    # solution map from model parameters: sol_map(p) -> x
    sol_map = nm.system.Node(func, ["p", "a"], ["x_bar"], name="smap")
    # penalty loss for mapping
    components = [sol_map]
    loss = nm.loss.PenaltyLoss(obj_bar, constrs_bar)
    problem = nm.problem.Problem(components, loss)
    # penalty loss for rounding
    components = [sol_map, round_module]
    loss = nm.loss.PenaltyLoss(obj_rnd, constrs_rnd)
    problem_rnd = nm.problem.Problem(components, loss)
    return problem, problem_rnd

def train():
    with wandb.init(resume=True, entity="botang") as run:
        # get config
        config = wandb.config
        print(config)
        # hyperparameters
        lr = config.learning_rate              # step size for gradient descent
        batch_size = config.batch_size         # batch size
        epochs = 400                           # number of training epochs
        warmup = 50                            # number of epochs to wait before enacting early stopping policy
        patience = 50                          # number of epochs with no improvement in eval metric to allow before early stopping
        hlayers_sol = config.hidden_layers_sol # number of hidden layers for solution mapping
        hsize_sol = config.hidden_size_sol     # width of hidden layers for solution mapping
        hlayers_rnd = config.hidden_layers_rnd # number of hidden layers for solution mapping
        hsize_rnd = config.hidden_size_rnd     # width of hidden layers for solution mapping
        # learning to round
        layers_rnd = netFC(input_dim=num_vars*2,
                           hidden_dims=[hsize_rnd]*hlayers_rnd,
                           output_dim=num_vars)
        round_func = roundModel(layers=layers_rnd, param_keys=["p", "a"],
                                var_keys=["x_bar"], output_keys=["x_rnd"],
                                int_ind={"x_bar":model.int_ind}, name="round")
        _, problem = getNMProb(round_func, hidden_dims=[hsize_sol]*hlayers_sol)
        # set adamW as optimizer
        optimizer = torch.optim.AdamW(problem.parameters(), lr=lr)
        # define trainer
        trainer = nm.trainer.Trainer(problem,
                                     loader_train, loader_dev, loader_test,
                                     optimizer, epochs=epochs,
                                     patience=patience, warmup=warmup)
        best_model = trainer.train()

        # eval
        sols, objvals, conviols, elapseds = [], [], [], []
        for p, a in tqdm(list(zip(p_test, a_test))):
            datapoints = {"p": torch.tensor(np.array([p]), dtype=torch.float32),
                          "a":torch.tensor(np.array([a]), dtype=torch.float32),
                          "name": "test"}
            tick = time.time()
            output = problem(datapoints)
            tock = time.time()
            x = output["test_x_rnd"]
            # get values
            model.setParamValue(p, *a)
            for ind in model.x:
                model.x[ind].value = x[0, ind].item()
            xval, objval = model.getVal()
            sols.append(xval.values())
            objvals.append(objval)
            conviols.append(sum(model.calViolation()))
            elapseds.append(tock - tick)
        df = pd.DataFrame({"Sol":sols, "Obj Val": objvals,
                           "Constraints Viol": conviols, "Elapsed Time": elapseds})
        time.sleep(1)
        print(df.describe())
        # log metrics to wandb
        wandb.log({"Mean Objective Value": df["Obj Val"].mean(),
                   "Mean Constraint Violation": df["Constraints Viol"].mean()})


if __name__ == "__main__":

    import wandb

    # sweep config
    sweep_config = {
        "name": "Rosenbrock-round",
        "method": "random",
        "metric": {
          "name": "Mean Objective Value",
          "goal": "minimize"
        },
        "parameters": {
            "learning_rate": {
                "min": 1e-5,
                "max": 1e-2
            },
            "batch_size": {
                "values": [16, 32, 64]
            },
            "hidden_layers_sol": {
                "values": [2, 3, 4, 5]
            },
            "hidden_size_sol": {
                "values": [20, 30, 40, 50]
            },
            "hidden_layers_rnd": {
                "values": [2, 3, 4, 5]
            },
            "hidden_size_rnd": {
                "values": [20, 40, 60, 80]
            }
        }
    }

    # init
    sweep_id = wandb.sweep(sweep_config,
                           project="Rosenbrock-round")

    # launch agent
    wandb.agent(sweep_id, function=train, count=50)
