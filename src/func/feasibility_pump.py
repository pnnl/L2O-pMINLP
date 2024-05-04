"""
Learnable feasibility pump with modularized rounding and projection
"""

import torch
from torch import nn

class feasibilityPumpModel(nn.Module):
    """
    Learnable feasibility pump model
    """
    def __init__(self, num_iters, nm_problem, round_module, round_layers,
                 project_module, project_layers, param_keys, var_keys,
                 output_keys=[], name="FeasibilityPump"):
        super(roundModel, self).__init__()
        # number of iterations
        self.num_iters = num_iters
        # learnable part
        self.problem = nm_problem
        self.round = round_module
        self.proj = project_module
        self.layers = self._get_fp_layers(self.num_iters, self.problem,
                                          self.round, self.proj)
        # data keys
        self.param_keys, self.var_keys = param_keys, var_keys
        self.input_keys = self.param_keys + self.var_keys
        self.output_keys = output_keys if output_keys else [self.var_key]
        # name
        self.name = name

    def _get_fp_layers(self, num_iters, problem, round, project)




if __name__ == "__main__":

    import numpy as np

    # random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # init
    num_data = 5000   # number of data
    test_size = 1000  # number of test size
    val_size = 1000   # number of validation size

    # data sample from uniform distribution
    p_low, p_high = 0.0, 1.0
    p_samples = torch.FloatTensor(num_data, 2).uniform_(p_low, p_high)
    data = {"p":p_samples}
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

    # get objective function & constraints
    from src.problem import nmQuadratic
