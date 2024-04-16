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

from src.problem import msQuadratic, nmQuadratic
from src.func.layer import netFC
from src.func import roundModel, roundGumbelModel, roundThresholdModel

def load_data(num_data, test_size, val_size):
    """
    Generate random data for training, testing, and validation.

    Args:
        num_data (int): Total number of data points.
        test_size (int): Size of the test dataset.
        val_size (int): Size of the validation dataset.

    Returns:
        tuple: Three DataLoader instances for training, testing, and validation datasets.
    """
    train_size = num_data - test_size - val_size
    # parameters as input data
    p_low, p_high = 0.0, 1.0
    p_train = np.random.uniform(p_low, p_high, (train_size, 2)).astype(np.float32)
    p_test  = np.random.uniform(p_low, p_high, (test_size, 2)).astype(np.float32)
    p_dev   = np.random.uniform(p_low, p_high, (val_size, 2)).astype(np.float32)
    # nm datasets
    data_train = DictDataset({"p":p_train}, name="train")
    data_test  = DictDataset({"p":p_test}, name="test")
    data_dev   = DictDataset({"p":p_dev}, name="dev")
    # torch dataloaders
    loader_train = DataLoader(data_train, batch_size=32, num_workers=0,
                              collate_fn=data_train.collate_fn, shuffle=True)
    loader_test  = DataLoader(data_test, batch_size=32, num_workers=0,
                              collate_fn=data_test.collate_fn, shuffle=False)
    loader_dev   = DataLoader(data_dev, batch_size=32, num_workers=0,
                              collate_fn=data_dev.collate_fn, shuffle=True)
    return loader_train, loader_test, loader_dev


def train(method_config):
    """
    Train models using the specified training method configuration.

    Args:
        method_config (object): Configuration containing details of method.
    """
    with wandb.init(resume=True, entity="botang") as run:
        # get config from wandb
        config = wandb.config
        print(config)
        # build problems for relaxation and rounding phases
        problem_rel, problem_rnd = build_problem(config, method_config)
        # 2-stage training, if specified
        if method_config.train == "2s":
            # get nm trainer
            trainer = get_trainer(config, problem_rel)
            # training for the relaxation problem
            best_model = trainer.train()
            # load best model dict
            problem_rel.load_state_dict(best_model)
            # free model parameters
            problem_rel.freeze()
        # get nm trainer
        trainer = get_trainer(config, problem_rnd)
        # training for the rounding problem
        best_model = trainer.train()
        # load best model dict
        problem_rnd.load_state_dict(best_model)
        # evaluate model
        df = eval(loader_test.dataset, problem_rnd)
        mean_obj_val = df["Obj Val"].mean()
        mean_constr_viol = df["Constraints Viol"].mean()
        mean_merit = mean_obj_val + 100 * mean_constr_viol
        wandb.log({"Mean Objective Value": mean_obj_val,
                   "Mean Constraint Violation": mean_constr_viol,
                   "Mean Merit": mean_merit})


def build_problem(config, method_config):
    """
    Build optimization problems using Neuromancer for both relaxation and rounding.

    Args:
        config (object): Configuration from wandb.
        method_config (object): Configuration containing details of method.

    Returns:
        tuple: Two neuromancer.problem.Problem instances for relaxation and rounding.
    """
    # hyperparameters
    penalty_weight = 100 #config.penalty_weight  # weight of constraint violation penealty
    hlayers_sol = config.hidden_layers_sol       # number of hidden layers for solution mapping
    hsize_sol = config.hidden_size_sol           # width of hidden layers for solution mapping
    hlayers_rnd = config.hidden_layers_rnd       # number of hidden layers for solution mapping
    hsize_rnd = config.hidden_size_rnd           # width of hidden layers for solution mapping
    continuous_update = config.continuous_update # update continuous variable during rounding step or not
    # define quadratic objective functions and constraints for both problem types
    obj_rel, constrs_rel = nmQuadratic(["x"], ["p"], penalty_weight=penalty_weight)
    obj_rnd, constrs_rnd = nmQuadratic(["x_rnd"], ["p"], penalty_weight=penalty_weight)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=2, outsize=4, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize_sol]*hlayers_sol)
    smap = nm.system.Node(func, ["p"], ["x"], name="smap")
    # select round method from method configuration
    round_methods = {
        "standard": roundModel,
        "gumbel": roundGumbelModel,
        "threshold": roundThresholdModel
    }
    rnd_class = round_methods[method_config.round]
    # define rounding model
    layers_rnd = netFC(input_dim=6, hidden_dims=[hsize_rnd]*hlayers_rnd, output_dim=4)
    rnd = rnd_class(layers=layers_rnd, param_keys=["p"], var_keys=["x"], output_keys=["x_rnd"],
                    int_ind={"x":[2,3]}, continuous_update=continuous_update, name="round")
    # build neuromancer problem for relaxation
    components = [smap]
    loss = nm.loss.PenaltyLoss(obj_rel, constrs_rel)
    problem_rel = nm.problem.Problem(components, loss)
    # build neuromancer problem for rounding
    components = [smap, rnd]
    loss = nm.loss.PenaltyLoss(obj_rnd, constrs_rnd)
    problem_rnd = nm.problem.Problem(components, loss)
    return problem_rel, problem_rnd


def get_trainer(config, problem):
    """
    Configure a trainer for a neuromancer problem.

    Args:
        config (object): Configuration from wandb.
        problem (nm.problem.Problem): Problem instance to train.

    Returns:
        nm.trainer.Trainer: Configured trainer.
    """
    # hyperparameters
    optim_type = config.optimizer    # type of optimizer
    lr = config.learning_rate       # step size for gradient descent
    batch_size = config.batch_size  # batch size
    epochs = 200                    # number of training epochs
    warmup = 20                     # number of epochs to wait before enacting early stopping policy
    patience = 20                   # number of epochs with no improvement in eval metric to allow before early stopping
    # set optimizer
    if optim_type == "SGD":
        optimizer = torch.optim.SGD(problem.parameters(), lr=lr)
    if optim_type == "Adam":
        optimizer = torch.optim.Adam(problem.parameters(), lr=lr)
    # create a trainer for the problem
    trainer = nm.trainer.Trainer(problem,
                                 loader_train, loader_dev, loader_test,
                                 optimizer, epochs=epochs,
                                 patience=patience, warmup=warmup)
    return trainer


def eval(dataset, problem):
    """
    Evaluate a trained model on a dataset.

    Args:
        dataset (neuromancer.dataset.DictDataset): Dataset for evaluation.
        problem (nm.problem.Problem): Trained problem model.

    Returns:
        pd.DataFrame: Results including solution, objective value, constraints violation, and elapsed time.
    """
    # exact mathmatical programming solver
    model = msQuadratic()
    # init
    sols, objvals, conviols, elapseds = [], [], [], []
    # iterate through test dataset
    for p in tqdm(dataset.datadict["p"]):
        datapoints = {"p": torch.tensor(np.array([p]), dtype=torch.float32), "name": "test"}
        # inference
        tick = time.time()
        output = problem(datapoints)
        tock = time.time()
        # get values
        x = output["test_x_rnd"]
        # assign params
        model.set_param_val({"p":p})
        # assign vars
        for i in model.vars["x"]:
            model.vars["x"][i].value = x[0, i].item()
        # get solutions
        xval, objval = model.get_val()
        # add results
        sols.append(xval.values())
        objvals.append(objval)
        conviols.append(sum(model.cal_violation()))
        elapseds.append(tock - tick)
    # create DataFrame
    df = pd.DataFrame({"Sol":sols, "Obj Val": objvals,
                       "Constraints Viol": conviols, "Elapsed Time": elapseds})
    time.sleep(1)
    # statistic
    print(df.describe())
    return df

if __name__ == "__main__":

    import argparse
    import wandb

    # turn off warning
    import logging
    logging.getLogger("pyomo.core").setLevel(logging.ERROR)

    # set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # argument parsing for command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--round",
                        type=str,
                        default="standard",
                        choices=["standard", "gumbel", "threshold"],
                        help="method for rounding")
    parser.add_argument("--train",
                        type=str,
                        default="e2e",
                        choices=["e2e", "2s"],
                        help="training pattern")
    # get configuration
    method_config = parser.parse_args()

    # configuration for sweep (hyperparameter tuning)
    sweep_config = {
        "name": "QP-round",
        "method": "bayes",
        "metric": {
          "name": "Mean Merit",
          "goal": "minimize"
        },
        "parameters": {
            #"penalty_weight":{
            #    "min": 0,
            #    "max": 500
            #},
            "optimizer": {
                "values": ["SGD", "Adam"]
            },
            "learning_rate": {
                "distribution": "log_uniform",
                "min": -5,  # 10^-5
                "max": -2   # 10^-2
            },
            "batch_size": {
                "values": [16, 32, 64]
            },
            "hidden_layers_sol": {
                "values": [1, 2, 3]
            },
            "hidden_size_sol": {
                "values": [4, 8, 16, 32]
            },
            "hidden_layers_rnd": {
                "values": [1, 2, 3]
            },
            "hidden_size_rnd": {
                "values": [4, 8, 16, 32]
            },
            "continuous_update": {
                "values": [False, True]
            }
        }
    }

    # load dataset
    num_data = 5000   # number of data
    test_size = 1000  # number of test size
    val_size = 1000   # number of validation size
    loader_train, loader_test, loader_dev = load_data(num_data, test_size, val_size)

    # set up project and sweep
    project_name = "QP-round-" + method_config.round + "-" + method_config.train
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, function=lambda: train(method_config), count=50)
