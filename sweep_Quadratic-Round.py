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
from src.problem.neuromancer.trainer import trainer

def load_data(num_ineq, num_data, test_size, val_size, batch_size):
    """
    Generate random data for training, testing, and validation.

    Args:
        num_ineq (int): number of inequality constraints.
        num_data (int): Total number of data points.
        test_size (int): Size of the test dataset.
        val_size (int): Size of the validation dataset.
        batch_size (int): Size of the mini-batch.

    Returns:
        tuple: Three DataLoader instances for training, testing, and validation datasets.
    """
    train_size = num_data - test_size - val_size
    # parameters as input data
    b_train = np.random.uniform(-1, 1, size=(train_size, num_ineq)).astype(np.float32)
    b_test  = np.random.uniform(-1, 1, size=(test_size, num_ineq)).astype(np.float32)
    b_dev   = np.random.uniform(-1, 1, size=(val_size, num_ineq)).astype(np.float32)
    # nm datasets
    data_train = DictDataset({"b":b_train}, name="train")
    data_test  = DictDataset({"b":b_test}, name="test")
    data_dev   = DictDataset({"b":b_dev}, name="dev")
    # torch dataloaders
    loader_train = DataLoader(data_train, batch_size=batch_size, num_workers=0,
                              collate_fn=data_train.collate_fn, shuffle=True)
    loader_test  = DataLoader(data_test, batch_size=batch_size, num_workers=0,
                              collate_fn=data_test.collate_fn, shuffle=False)
    loader_dev   = DataLoader(data_dev, batch_size=batch_size, num_workers=0,
                              collate_fn=data_dev.collate_fn, shuffle=True)
    return loader_train, loader_test, loader_dev


def train(method_config):
    """
    Train models using the specified training method configuration.

    Args:
        method_config (object): Configuration containing details of method.
    """
    num_var = method_config.num_var      # number of variable
    num_ineq = method_config.num_ineq    # number of inequality
    with wandb.init(resume=True, entity="botang") as run:
        # get config from wandb
        config = wandb.config
        print(config)
        # load data
        loader_train, loader_test, loader_dev = load_data(num_ineq, num_data,
                                                          test_size, val_size,
                                                          batch_size=config.batch_size)
        # build problems for relaxation and rounding phases
        components, loss_fn = build_problem(config, method_config)
        # 2-stage training, if specified
        if method_config.train == "2s":
            # get nm trainer
            trainer = get_trainer(config, nn.ModuleList([components[0]]), loss_fn)
            # training for the relaxation problem
            best_model = trainer.train(loader_train, loader_test)
            # free model parameters
            components[0].freeze()
        # get nm trainer
        trainer = get_trainer(config, components, loss_fn)
        # training for the rounding problem
        best_model = trainer.train(loader_train, loader_test)
        # evaluate model
        df = eval(loader_test.dataset, components, num_var)
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
    # problem size
    num_var = method_config.num_var      # number of variable
    num_ineq = method_config.num_ineq    # number of inequality
    # hyperparameters
    penalty_weight = config.penalty_weight       # weight of constraint violation penealty
    hlayers_sol = config.hidden_layers_sol       # number of hidden layers for solution mapping
    hlayers_rnd = config.hidden_layers_rnd       # number of hidden layers for solution mapping
    hsize = config.hidden_size                   # width of hidden layers for solution mapping
    continuous_update = config.continuous_update # update continuous variable during rounding step or not
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_ineq, outsize=num_var, bias=True,
                             linear_map=nm.slim.maps["linear"],
                             nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["b"], ["x"], name="smap")
    # select round method from method configuration
    round_methods = {
        "standard": roundModel,
        "gumbel": roundGumbelModel,
        "threshold": roundThresholdModel
    }
    rnd_class = round_methods[method_config.round]
    # define rounding model
    layers_rnd = netFC(input_dim=num_ineq+num_var,
                       hidden_dims=[hsize]*hlayers_rnd,
                       output_dim=num_var)
    rnd = rnd_class(layers=layers_rnd, param_keys=["b"], var_keys=["x"],
                    output_keys=["x_rnd"], int_ind={"x":range(num_var)},
                    continuous_update=continuous_update, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd])
    # loss function
    loss_fn = nmQuadratic(["b", "x_rnd"], Q, p, A, penalty_weight)
    return components, loss_fn


def get_trainer(config, components, loss_fn):
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
    epochs = 400                    # number of training epochs
    warmup = 40                     # number of epochs to wait before enacting early stopping policy
    patience = 40                   # number of epochs with no improvement in eval metric to allow before early stopping
    # set optimizer
    optimizers = {"SGD": torch.optim.SGD,
                  "Adam": torch.optim.Adam,
                  "AdamW": torch.optim.AdamW
    }
    optimizer = optimizers[optim_type](components.parameters(), lr=lr)
    # create a trainer for the problem
    my_trainer = trainer(components, loss_fn, optimizer, epochs, patience, warmup)
    return my_trainer


def eval(dataset, components, num_var):
    """
    Evaluate a trained model on a dataset.

    Args:
        dataset (neuromancer.dataset.DictDataset): Dataset for evaluation.
        problem (nm.problem.Problem): Trained problem model.
        num_var (int): number of varibles

    Returns:
        pd.DataFrame: Results including solution, objective value, constraints violation, and elapsed time.
    """
    # exact mathmatical programming solver
    model = msQuadratic(Q.cpu().numpy(), p.cpu().numpy(), A.cpu().numpy())
    # init
    sols, objvals, conviols, elapseds = [], [], [], []
    # iterate through test dataset
    for b in tqdm(dataset.datadict["b"]):
        datapoints = {"b": torch.FloatTensor(np.array([b])),
                      "name": "test"}
        # inference
        tick = time.time()
        with torch.no_grad():
            for comp in components:
                datapoints.update(comp(datapoints))
        tock = time.time()
        # get values
        x = datapoints["x_rnd"]
        # assign params
        model.set_param_val({"b":b})
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
    parser.add_argument("--num_var",
                        type=int,
                        default=50,
                        help="number of varibles")
    parser.add_argument("--num_ineq",
                        type=int,
                        default=50,
                        help="number of inequality constraints")
    # get configuration
    method_config = parser.parse_args()

    # configuration for sweep (hyperparameter tuning)
    sweep_config = {
        "name": "Rosenbrock-round",
        "method": "bayes",
        "metric": {
          "name": "Mean Merit",
          "goal": "minimize"
        },
        "parameters": {
            "penalty_weight":{
                "min": 10,
                "max": 200
            },
            "optimizer": {
                "values": ["AdamW"]
            },
            "learning_rate": {
                "distribution": "q_log_uniform_values",
                "q": 1e-5,
                "min": 1e-5,  # 10^-5
                "max": 1e0    # 10^0
            },
            "batch_size": {
                "values": [64, 128, 256]
            },
            "hidden_layers_sol": {
                "values": [2, 3, 4, 5]
            },
            "hidden_layers_rnd": {
                "values": [2, 3, 4, 5]
            },
            "hidden_size": {
                "values": [32, 64, 128, 256]
            },
            "continuous_update": {
                "values": [False, True]
            }
        }
    }

    # size
    num_var = method_config.num_var      # number of variable
    num_ineq = method_config.num_ineq    # number of inequality
    num_data = 5000                      # number of data
    test_size = 1000                     # number of test size
    val_size = 1000                      # number of validation size

    # generate fixed parameters
    np.random.seed(17)
    # generate parameters
    Q = torch.from_numpy(np.diag(np.random.random(size=num_var))).float()
    p = torch.from_numpy(np.random.random(num_var)).float()
    A = torch.from_numpy(np.random.normal(scale=1, size=(num_ineq, num_var))).float()

    # set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # set up project and sweep
    project_name = "Quadratic-{}-{}-{}".format(method_config.num_var,
                                               method_config.num_ineq,
                                               method_config.train)
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, function=lambda: train(method_config), count=100)
