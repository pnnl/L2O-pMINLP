import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import neuromancer as nm
from tqdm import tqdm

from src.problem import nmNonconvex, msNonconvex
from src.func.layer import netFC
from src.func import roundGumbelModel, roundThresholdModel

# random seed
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def set_components(method, num_var, num_ineq, hlayers_sol, hlayers_rnd, hwidth):
    """
    Set components for NN model with rounding correction
    """
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_ineq, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hwidth]*hlayers_sol)
    smap = nm.system.Node(func, ["b"], ["x"], name="smap")
    # define rounding model
    layers_rnd = netFC(input_dim=num_ineq+num_var, hidden_dims=[hwidth]*hlayers_rnd, output_dim=num_var)
    if method == "classfication":
        rnd = roundThresholdModel(layers=layers_rnd, param_keys=["b"], var_keys=["x"], output_keys=["x_rnd"],
                                  int_ind={"x":range(num_var)}, continuous_update=True, name="round")
    else:
        rnd = roundGumbelModel(layers=layers_rnd, param_keys=["b"], var_keys=["x"], output_keys=["x_rnd"],
                               int_ind={"x":range(num_var)}, continuous_update=True, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    return components


def eval(data_test, model, components):
    """
    Evaluate model performence
    """
    params, sols, objvals, conviols, elapseds = [], [], [], [], []
    for b in tqdm(data_test.datadict["b"][:100]):
        # data point as tensor
        datapoints = {"b": torch.unsqueeze(b, 0).to("cuda"),
                      "name": "test"}
        # infer
        components.eval()
        tick = time.time()
        with torch.no_grad():
            for comp in components:
                datapoints.update(comp(datapoints))
        tock = time.time()
        # assign params
        model.set_param_val({"b":b.cpu().numpy()})
        # assign vars
        x = datapoints["x_rnd"]
        for i in range(num_var):
            model.vars["x"][i].value = x[0,i].item()
        # get solutions
        xval, objval = model.get_val()
        params.append(list(b.cpu().numpy()))
        sols.append(list(list(xval.values())[0].values()))
        objvals.append(objval)
        conviols.append(sum(model.cal_violation()))
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param":params, "Sol":sols, "Obj Val": objvals, "Constraints Viol": conviols, "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solution: {}".format(np.sum(df["Constraints Viol"] > 0)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # experiments configuration
    parser.add_argument("--round",
                        type=str,
                        default="classfication",
                        choices=["classfication", "threshold"],
                        help="method for rounding")
    parser.add_argument("--size",
                        type=int,
                        default=50,
                        choices = [5, 10, 20, 50, 100, 200, 500],
                        help="number of decsion variables and constraints")
    # get experiment setting
    config = parser.parse_args()

    # init
    num_var = config.size     # number of variables
    num_ineq = config.size    # number of constraints
    num_data = 10000          # number of data
    test_size = 1000          # number of test size
    val_size = 1000           # number of validation size
    train_size = num_data - test_size - val_size

    # data sample from uniform distribution
    b_samples = torch.from_numpy(np.random.uniform(-1, 1, size=(num_data, num_ineq))).float()
    data = {"b":b_samples}
    # data split
    from src.utlis import data_split
    data_train, data_test, data_dev = data_split(data, test_size=test_size, val_size=val_size)

    # torch dataloaders
    from torch.utils.data import DataLoader
    batch_size = 64
    loader_train = DataLoader(data_train, batch_size, num_workers=0,
                              collate_fn=data_train.collate_fn, shuffle=True)
    loader_test  = DataLoader(data_test, batch_size, num_workers=0,
                              collate_fn=data_test.collate_fn, shuffle=False)
    loader_dev   = DataLoader(data_dev, batch_size,
                              num_workers=0, collate_fn=data_dev.collate_fn, shuffle=False)

    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # hyperparameters
    hsize_dict = {5:16, 10:32, 20:64, 50:128, 100:256, 200:512, 500:1024}
    penalty_weight = 100          # weight of constraint violation penealty
    hlayers_sol = 5               # number of hidden layers for solution mapping
    hlayers_rnd = 4               # number of hidden layers for solution mapping
    hwidth = hsize_dict[num_var]  # width of hidden layers for solution mapping
    lr = 1e-3                     # learning rate

    # get components
    components = set_components(config.round, num_var, num_ineq, hlayers_sol, hlayers_rnd, hwidth)
    # loss function with constraint penalty
    loss_fn = nmNonconvex(["b", "x_rnd"], num_var, num_ineq, penalty_weight=100)

    # training
    from src.problem.neuromancer.trainer import trainer
    epochs = 200                    # number of training epochs
    warmup = 20                     # number of epochs to wait before enacting early stopping policy
    patience = 20                   # number of epochs with no improvement in eval metric to allow before early stopping
    optimizer = torch.optim.AdamW(components.parameters(), lr=lr)
    # create a trainer for the problem
    my_trainer = trainer(components, loss_fn, optimizer, epochs, patience, warmup, device="cuda")
    # training for the rounding problem
    my_trainer.train(loader_train, loader_dev)

    # eval
    model = msNonconvex(num_var, num_ineq)
    eval(data_test, model, components)
