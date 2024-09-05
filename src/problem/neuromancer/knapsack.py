"""
Parametric Multi-Dimensional Knapsack
"""

import numpy as np
import torch
from torch import nn
import neuromancer as nm


class penaltyLoss(nn.Module):
    """
    Penalty loss function for knapsack problem
    """
    def __init__(self, input_keys, weights, caps, penalty_weight=50, output_key="loss"):
        super().__init__()
        self.c_key, self.x_key = input_keys
        self.output_key = output_key
        self.weights = torch.from_numpy(weights).to(torch.float32)
        self.caps = torch.tensor(caps).to(torch.float32)
        self.penalty_weight = penalty_weight

    def forward(self, input_dict):
        """
        forward pass
        """
        # objective function
        obj = self.cal_obj(input_dict)
        # constraints violation
        viol = self.cal_constr_viol(input_dict)
        # penalized loss
        loss = obj + self.penalty_weight * viol
        input_dict[self.output_key] = torch.mean(loss)
        return input_dict

    def cal_obj(self, input_dict):
        """
        calculate objective function
        """
        # get values
        x, c = input_dict[self.x_key], input_dict[self.c_key]
        # objective function (maximize)
        f = - torch.einsum("bn,bn->b", c, x)
        return f

    def cal_constr_viol(self, input_dict):
        """
        calculate constraints violation
        """
        # get values
        x = input_dict[self.x_key]
        # capacity constraints
        lhs = torch.einsum("bj,ij->bi", x, self.weights)
        violation = torch.relu(lhs - self.caps).sum(dim=1)
        # non-negative constraints
        violation += torch.relu(-x).sum(dim=1)
        return violation


if __name__ == "__main__":

    # random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # init
    num_var = 20      # number of variables
    dim = 2           # dimension of constraints
    caps = [20] * dim # capacity
    num_data = 5000   # number of data
    test_size = 1000  # number of test size
    val_size = 1000   # number of validation size

    # data sample from PyEPO
    import pyepo
    # generate data
    weights, x, c = pyepo.data.knapsack.genData(num_data=num_data, num_features=5,
                                                num_items=num_var, dim=dim,
                                                deg=4, noise_width=0.5)
    c_samples = torch.FloatTensor(c)
    data = {"c":c_samples}
    # data split
    from src.utlis import data_split
    data_train, data_test, data_dev = data_split(data, test_size=test_size, val_size=val_size)
    # torch dataloaders
    from torch.utils.data import DataLoader
    loader_train = DataLoader(data_train, batch_size=32, num_workers=0,
                              collate_fn=data_train.collate_fn, shuffle=True)
    loader_test  = DataLoader(data_test, batch_size=32, num_workers=0,
                              collate_fn=data_test.collate_fn, shuffle=True)
    loader_dev   = DataLoader(data_dev, batch_size=32, num_workers=0,
                              collate_fn=data_dev.collate_fn, shuffle=True)

    # define neural architecture for the solution map smap(c) -> x
    import neuromancer as nm
    func = nm.modules.blocks.MLP(insize=num_var, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[64]*2)
    components =  nn.ModuleList([nm.system.Node(func, ["c"], ["x"], name="smap")])

    # build neuromancer problems
    loss_fn = penaltyLoss(["c", "x"], weights, caps)

    # training
    lr = 0.001    # step size for gradient descent
    epochs = 400  # number of training epochs
    warmup = 40   # number of epochs to wait before enacting early stopping policy
    patience = 40 # number of epochs with no improvement in eval metric to allow before early stopping
    # set adamW as optimizer
    optimizer = torch.optim.AdamW(components.parameters(), lr=lr)
    # training
    from src.problem.neuromancer.trainer import trainer
    my_trainer = trainer(components, loss_fn, optimizer, epochs, patience, warmup)
    my_trainer.train(loader_train, loader_dev)
    print()

    # init mathmatic model
    from src.problem.math_solver.knapsack import knapsack
    model = knapsack(weights=weights, caps=caps)

    # test neuroMANCER
    from src.utlis import nm_test_solve
    c = c[0]
    datapoint = {"c": torch.tensor([c], dtype=torch.float32),
                 "name":"test"}
    model.set_param_val({"c":c})
    print("neuroMANCER:")
    nm_test_solve("x", components, datapoint, model)
