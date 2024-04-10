"""
Learning to round
"""
from collections import defaultdict

import torch
from torch import nn

from src.func.ste import diffFloor, diffBinarize, diffGumbelBinarize, thresholdBinarize

class roundModel(nn.Module):
    """
    Learnable model to round integer variables
    """
    def __init__(self, layers, param_keys, var_keys, output_keys=[],
                 int_ind=defaultdict(list), bin_ind=defaultdict(list),
                 continuous_update=False, tolerance=1e-3, name=None):
        super(roundModel, self).__init__()
        # data keys
        self.param_keys, self.var_keys = param_keys, var_keys
        self.input_keys = self.param_keys + self.var_keys
        self.output_keys = output_keys if output_keys else self.var_keys
        # index of integer and binary variables
        self.int_ind = int_ind
        self.bin_ind = bin_ind
        # update continuous or not
        self.continuous_update = continuous_update
        # numerical tolerance
        self.tolerance = tolerance
        # autograd functions
        self.floor, self.bin = diffFloor(), diffBinarize()
        # sequence
        self.layers = layers
        # name
        self.name = name

    def forward(self, data):
        # get vars & params
        p, x = self._extractData(data)
        # concatenate all features: params + sol
        f = torch.cat(p+x, dim=-1)
        # forward
        h = self.layers(f)
        # rounding
        output_data = self._processRounding(h, data)
        return output_data

    def _extractData(self, data):
        p = [data[k] for k in self.param_keys]
        x = [data[k] for k in self.var_keys]
        return p, x

    def _processRounding(self, h, data):
        output_data = {}
        for k_in, k_out in zip(self.var_keys, self.output_keys):
            # get rounding
            x_rnd = self._roundVars(h, data, k_in)
            output_data[k_out] = x_rnd
            # cut off used h
            h = h[:,data[k_in].shape[1]+1:]
        return output_data

    def _roundVars(self, h, data, key):
        # get index
        int_ind = self.int_ind[key]
        bin_ind = self.bin_ind[key]
        ###################### integer ######################
        # floor(x)
        x_flr = self.floor(data[key][:,int_ind])
        # bin(h): binary 0 for floor, 1 for ceil
        bnr = self.bin(h[:,int_ind])
        # mask if already integer
        bnr = self._intMask(bnr, data[key][:, int_ind])
        # update continuous variables or not
        if self.continuous_update:
            x_rnd = data[key] + h
        else:
            x_rnd = data[key].clone()
        # update rounding for integer variables int(x) = floor(x) + bin(h)
        x_rnd[:, int_ind] = x_flr + bnr
        ###################### binary ######################
        # update rounding for binary variables: bin(x) = bin(h)
        x_rnd[:, bin_ind] = self.bin(h[:, bin_ind])
        return x_rnd

    def _intMask(self, bnr, x):
        # difference
        diff_fl = x - torch.floor(x)
        diff_cl = torch.ceil(x) - x
        # mask
        bnr[diff_fl < self.tolerance] = 0.0
        bnr[diff_cl < self.tolerance] = 1.0
        return bnr

    def freeze(self):
        """
        Freezes the parameters of the callable in this node
        """
        for param in self.layers.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreezes the parameters of the callable in this node
        """
        for param in self.layers.parameters():
            param.requires_grad = True


class roundGumbelModel(roundModel):
    """
    Learnable model to round integer variables with Gumbel-Softmax trick
    """
    def __init__(self, layers,param_keys, var_keys, output_keys=[],
                 int_ind=defaultdict(list), bin_ind=defaultdict(list),
                 continuous_update=False, temperature=1.0, tolerance=1e-3, name=None):
        super(roundGumbelModel, self).__init__(layers, param_keys, var_keys,
                                               output_keys, int_ind, bin_ind,
                                               continuous_update, tolerance, name)
        # random temperature
        self.temperature = temperature
        # binarize
        self.bin = diffGumbelBinarize(temperature=self.temperature)


class roundThresholdModel(roundModel):
    """
    Learnable model to round integer variables with variable threshold
    """
    def __init__(self, layers, param_keys, var_keys,output_keys=[],
                 int_ind=defaultdict(list), bin_ind=defaultdict(list),
                 continuous_update=False, slope=10, name=None):
        super(roundThresholdModel, self).__init__(layers, param_keys, var_keys,
                                                  output_keys, int_ind, bin_ind,
                                                  continuous_update,
                                                  tolerance=None, name=name)
        # slope
        self.slope= slope
        # binarize
        self.bin = thresholdBinarize(slope=self.slope)

    def _roundVars(self, h, data, key):
        # get index
        int_ind = self.int_ind[key]
        bin_ind = self.bin_ind[key]
        # get threshold from sigmoid
        threshold = torch.sigmoid(h)
        ###################### integer ######################
        # floor(x)
        x_flr = self.floor(data[key][:,int_ind])
        # extract fractional part
        x_frc = data[key][:,int_ind] - x_flr
        # get threshold
        v = threshold[:,int_ind]
        # bin(x, v): binary 0 for floor, 1 for ceil
        bnr = self.bin(x_frc, v)
        # update continuous variables or not
        if self.continuous_update:
            x_rnd = data[key] + h
        else:
            x_rnd = data[key].clone()
        # update rounding for integer variables int(x) = floor(x) + bin(h)
        x_rnd[:, int_ind] = x_flr + bnr
        ###################### binary ######################
        # get fractional variables
        x = data[key][:,bin_ind]
        # get threshold
        v = threshold[:,bin_ind]
        # bin(x,v): binary 0 for 0, 1 for c1
        bnr = self.bin(x, v)
        # update rounding for binary variables int(x) = bin(x, v)
        x_rnd[:, bin_ind] = bnr
        return x_rnd


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
    from src.utlis import dataSplit
    data_train, data_test, data_dev = dataSplit(data, test_size=test_size, val_size=val_size)
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
    obj, constrs = nmQuadratic(["x"], ["p"], penalty_weight=100)

    # define neural architecture for the solution map
    import neuromancer as nm
    func = nm.modules.blocks.MLP(insize=2, outsize=4, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[10]*4)
    smap = nm.system.Node(func, ["p"], ["x"], name="smap")

    # define rounding model
    from src.func.layer import netFC
    layers_rnd = netFC(input_dim=6, hidden_dims=[20]*3, output_dim=4)
    #round_func = roundModel(layers=layers_rnd, param_keys=["p"], var_keys=["x"], output_keys=["x_rnd"],
    #                        int_ind={"x":[2,3]}, continuous_update=False, name="round")
    round_func = roundGumbelModel(layers=layers_rnd, param_keys=["p"], var_keys=["x"], output_keys=["x_rnd"],
                                  bin_ind={"x":[2,3]}, continuous_update=False, name="round")
    #round_func = roundThresholdModel(layers=layers_rnd, param_keys=["p"], var_keys=["x"], output_keys=["x_rnd"],
    #                                 int_ind={"x":[2,3]}, continuous_update=False, name="round")


    # build neuromancer problem
    components = [smap, round_func]
    loss = nm.loss.PenaltyLoss(obj, constrs)
    problem = nm.problem.Problem(components, loss)

    # training
    lr = 0.001    # step size for gradient descent
    epochs = 20   # number of training epochs
    warmup = 50   # number of epochs to wait before enacting early stopping policy
    patience = 50 # number of epochs with no improvement in eval metric to allow before early stopping
    # set adamW as optimizer
    optimizer = torch.optim.AdamW(problem.parameters(), lr=lr)
    # define trainer
    trainer = nm.trainer.Trainer(
        problem,
        loader_train,
        loader_dev,
        loader_test,
        optimizer,
        epochs=epochs,
        patience=patience,
        warmup=warmup)
    # train solution map
    best_model = trainer.train()
    print()

    # init mathmatic model
    from src.problem.math_solver.quadratic import quadratic
    model = quadratic()

    # test neuroMANCER
    from src.utlis import nmSolveTest
    print("neuroMANCER:")
    datapoint = {"p": torch.tensor([[0.6, 0.8]], dtype=torch.float32),
                 "name":"test"}
    nmSolveTest(["x_rnd"], problem, datapoint, model)
