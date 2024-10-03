import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import neuromancer as nm
from tqdm import tqdm

from src.problem import nmRosenbrock, msRosenbrock
from src.func.layer import netFC
from src.func import roundGumbelModel, roundThresholdModel

# random seed
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def set_components(method, num_blocks, hlayers_sol, hlayers_rnd, hwidth):
    """
    Set components for NN model with rounding correction
    """
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_blocks+1, outsize=2*num_blocks, bias=True,
                             linear_map=nm.slim.maps["linear"],
                             nonlin=nn.ReLU, hsizes=[hwidth]*hlayers_sol)
    smap = nm.system.Node(func, ["p", "a"], ["x"], name="smap")
    # define rounding model
    layers_rnd = netFC(input_dim=3*num_blocks+1, hidden_dims=[hwidth]*hlayers_rnd, output_dim=2*num_blocks)
    if method == "classfication":
        rnd = roundThresholdModel(layers=layers_rnd, param_keys=["p", "a"], var_keys=["x"],  output_keys=["x_rnd"],
                                  int_ind={"x":[2*i+1 for i in range(num_blocks)]}, continuous_update=True, name="round")
    else:
        rnd = roundGumbelModel(layers=layers_rnd, param_keys=["b"], var_keys=["x"], output_keys=["x_rnd"],
                               int_ind={"x":[2*i+1 for i in range(num_blocks)]}, continuous_update=True, name="round")
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
                        default=100,
                        choices = [1, 10, 100, 1000, 10000],
                        help="number of decsion variables")
    # get experiment setting
    config = parser.parse_args()

    # init
    num_blocks = config.size  # number of blocks
    steepness = 50            # steepness of objective function
    num_data = 91000          # number of data
    test_size = 100           # number of test size
    val_size = 1000           # number of validation size
    train_size = num_data - test_size - val_size

    # parameters as input data
    p_low, p_high = 1.0, 8.0
    a_low, a_high = 0.5, 4.5
    p_train = np.random.uniform(p_low, p_high, (train_size, 1)).astype(np.float32)
    p_test  = np.random.uniform(p_low, p_high, (test_size, 1)).astype(np.float32)
    p_dev   = np.random.uniform(p_low, p_high, (val_size, 1)).astype(np.float32)
    a_train = np.random.uniform(a_low, a_high, (train_size, num_blocks)).astype(np.float32)
    a_test  = np.random.uniform(a_low, a_high, (test_size, num_blocks)).astype(np.float32)
    a_dev   = np.random.uniform(a_low, a_high, (val_size, num_blocks)).astype(np.float32)
    # nm datasets
    from neuromancer.dataset import DictDataset
    data_train = DictDataset({"p":p_train, "a":a_train}, name="train")
    data_test = DictDataset({"p":p_test, "a":a_test}, name="test")
    data_dev = DictDataset({"p":p_dev, "a":a_dev}, name="dev")
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
    hsize_dict = {1:4, 10:16, 100:64, 1000:256, 10000:1024}
    penalty_weight = 100              # weight of constraint violation penealty
    hlayers_sol = 5                  # number of hidden layers for solution mapping
    hlayers_rnd = 4                  # number of hidden layers for solution mapping
    hwidth = hsize_dict[num_blocks]  # width of hidden layers for solution mapping
    lr = 1e-3                        # learning rate

    # get components
    components = set_components(config.round, num_blocks, hlayers_sol, hlayers_rnd, hwidth)
    # loss function with constraint penalty
    loss_fn = nmRosenbrock(["p", "a", "x_rnd"], steepness, num_blocks, penalty_weight)

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
    model = msRosenbrock(steepness, num_blocks)
    eval(data_test, model, components)
