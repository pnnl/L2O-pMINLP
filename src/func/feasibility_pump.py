"""
Learnable feasibility pump with modularized rounding and projection
"""

import torch
from torch import nn

from src.func.rnd import roundGumbelModel

class feasibilityPumpModel:
    """
    Learnable feasibility pump model
    """
    def __init__(self, num_iters, nm_problem, penalty_weight, sol_map,
                 round_layers, proj_layers, param_keys, var_keys, output_keys=[],
                 name="FeasibilityPump"):
        super(feasibilityPumpModel, self).__init__()
        # number of iterations
        self.num_iters = num_iters
        # function to build nm problem
        self.problem = nm_problem
        # weight of Lagrangian penalty
        self.penalty_weight = penalty_weight
        # data keys
        self.param_keys, self.var_keys = param_keys, var_keys
        self.input_keys = self.param_keys + self.var_keys
        self.output_keys = output_keys if output_keys else [self.var_key]
        # name
        self.name = name
        # get components
        self.layers = self._get_fp_layers(sol_map, round_layers, proj_layers)

    def _get_fp_layers(self, sol_map, round_layers, project_layers):
        loss = 0
        components = [smap]
        for i in range(self.num_iters):
            loss, comp = self._construct_round(round_layers, i)

    def _construct_round(self, round_layers, ind):
        # build penalty loss from nm problem
        obj, constrs = self.problem(["x_rnd_{}".format(ind)], self.param_keys, self.penalty_weight)
        loss = nm.loss.PenaltyLoss(obj, constrs)
        # build rounding component
        roundGumbelModel(layers=round_layers, param_keys=["p"], var_keys=["x"], output_keys=["x_rnd"],
                        bin_ind={"x":[2,3]}, continuous_update=False, name="round")

    def _construct_project(self, proj_layers, ind):
        obj, constrs = self.problem(["x_proj_{}".format(ind)], self.param_keys, self.penalty_weight)


if __name__ == "__main__":

    import numpy as np

    # random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # init
    steepness = 30    # steepness factor
    num_blocks = 3    # number of expression blocks
    num_data = 5000   # number of data
    test_size = 1000  # number of test size
    val_size = 1000   # number of validation size

    # hyperparameters
    num_iters = 10        # number of feasibility pump iterations
    penalty_weight = 100  # weight of constraint violation penealty
    hlayers_sol = 4       # number of hidden layers for solution mapping
    hlayers_rnd = 4       # number of hidden layers for solution mapping
    hsize = 32            # width of hidden layers for solution mapping
    lr = 1e-2             # learning rate
    batch_size = 64       # batch size

    # data sample from uniform distribution
    p_low, p_high = 1.0, 8.0
    a_low, a_high = 0.5, 4.5
    p_samples = torch.FloatTensor(num_data, 1).uniform_(p_low, p_high)
    a_samples = torch.FloatTensor(num_data, num_blocks).uniform_(a_low, a_high)
    data = {"p":p_samples, "a":a_samples}
    # data split
    from src.utlis import data_split
    data_train, data_test, data_dev = data_split(data, test_size=test_size, val_size=val_size)
    # torch dataloaders
    from torch.utils.data import DataLoader
    loader_train = DataLoader(data_train, batch_size=batch_size, num_workers=0,
                              collate_fn=data_train.collate_fn, shuffle=True)
    loader_test  = DataLoader(data_test, batch_size=batch_size, num_workers=0,
                              collate_fn=data_test.collate_fn, shuffle=True)
    loader_dev   = DataLoader(data_dev, batch_size=batch_size, num_workers=0,
                              collate_fn=data_dev.collate_fn, shuffle=True)

    # get solution map
    import neuromancer as nm
    func = nm.modules.blocks.MLP(insize=num_blocks+1, outsize=2*num_blocks, bias=True,
                             linear_map=nm.slim.maps["linear"],
                             nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["p", "a"], ["x_init"], name="smap")

    # get nm problem
    from src.problem import nmRosenbrock
    my_nmRosenbrock = lambda var_keys, param_keys, penalty_weight: \
                      nmRosenbrock(var_keys, param_keys, steepness, num_blocks,
                                   penalty_weight=penalty_weight)

    # get rounding and projection layers
    from src.func.layer import netFC
    layers_rnd = netFC(input_dim=3*num_blocks+1, hidden_dims=[hsize]*hlayers_rnd,
                       output_dim=2*num_blocks)
    layers_proj = netFC(input_dim=10, hidden_dims=[20]*3, output_dim=4)

    fp_layers = feasibilityPumpModel(num_iters, my_nmRosenbrock, penalty_weight,
                                     smap, layers_rnd, layers_proj,
                                     param_keys=["p", "a"],
                                     var_keys=["x_init"],
                                     output_keys=["x"],
                                     name="FeasibilityPump")
