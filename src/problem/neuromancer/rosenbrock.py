"""
Parametric Mixed Integer Rosenbrock Problem
"""

import numpy as np
from scipy.linalg import null_space
import torch
from torch import nn
import neuromancer as nm

class penaltyLoss(nn.Module):
    """
    Penalty loss function for Rosenbrock problem
    """
    def __init__(self, input_keys, steepness, num_blocks, penalty_weight=50, output_key="loss"):
        super().__init__()
        self.b_key, self.a_key, self.x_key = input_keys
        self.output_key = output_key
        self.steepness = steepness
        self.num_blocks = num_blocks
        self.penalty_weight = penalty_weight
        self.device = None
        # coefs
        rng = np.random.RandomState(17)
        P = rng.normal(scale=1, size=(3, num_blocks))
        Q = rng.normal(scale=1, size=(3, num_blocks))
        self.P = torch.from_numpy(P).float()
        self.Q = torch.from_numpy(Q).float()

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
        x, a = input_dict[self.x_key], input_dict[self.a_key]
        # x_2i
        x1 = x[:, ::2]
        # x_2i+1
        x2 = x[:, 1::2]
        # objective function
        f = torch.sum((a - x1) ** 2 + self.steepness * (x2 - x1 ** 2) ** 2, dim=1)
        return f

    def cal_constr_viol(self, input_dict):
        """
        calculate constraints violation
        """
        # get values
        x, b = input_dict[self.x_key], input_dict[self.b_key]
        # update device
        if self.device is None:
            self.device = x.device
            self.P = self.P.to(self.device)
            self.Q = self.Q.to(self.device)
        # inner constraint violation
        lhs_inner = torch.sum(x[:, ::2] ** 2, dim=1)
        rhs_inner = self.num_blocks * b[:, 0] / 2
        inner_violation = torch.relu(rhs_inner - lhs_inner)
        # outer constraint violation
        lhs_outer = torch.sum(x[:, 1::2], dim=1)
        rhs_outer = self.num_blocks * b[:, 0]
        outer_violation = torch.relu(lhs_outer - rhs_outer)
        # lear constraint violation
        linear_violation = 0
        for i in range(3):
            lhs = torch.matmul(x[:, 0::2], self.P[i])
            linear_violation += torch.abs(lhs)
            lhs = torch.matmul(x[:, 1::2], self.Q[i])
            linear_violation += torch.relu(lhs)
        return inner_violation + outer_violation + linear_violation


class equalityEncoding(nn.Module):
    def __init__(self, num_blocks, input_key, output_key):
        """
        encode equality constraints P x = 0 using null space decomposition.
        """
        super().__init__()
        # size
        self.num_blocks = num_blocks
        # data keys
        self.input_key = input_key
        self.output_key = output_key
        # init encoding
        rng = np.random.RandomState(17)
        P = rng.normal(scale=1, size=(3, self.num_blocks))
        # sepecial solution for equality constraints
        x_s, _, _, _ = np.linalg.lstsq(P, np.zeros(3), rcond=None)
        # null space for equality constraints
        N = null_space(P)
        # to pytorch
        self.x_s = torch.tensor(x_s, dtype=torch.float32).view(1, -1)
        self.N = torch.tensor(N, dtype=torch.float32)
        # init device
        self.device = None

    def forward(self, data):
        # get free parameters
        z = data[self.input_key]
        # batch size
        batch_size = z.shape[0]
        # device
        if z.device != self.device:
            self.device = z.device
            self.x_s = self.x_s.to(self.device)
            self.N = self.N.to(self.device)
        # init x
        x = torch.zeros((batch_size, self.num_blocks*2)).to(self.device)
        # integer part
        x[:, 1::2] = z[:,self.num_blocks-3:]
        # continous part  to encode
        x[:, 0::2] = self.x_s + torch.einsum("bj,ij->bi", z[:,:self.num_blocks-3], self.N)
        data[self.output_key] = x
        # cut off z
        data[self.input_key] = z[:,:self.num_blocks-3]
        return data


if __name__ == "__main__":

    # random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # init
    steepness = 50
    num_blocks = 10
    hlayers_sol = 5
    hlayers_rnd = 4
    hsize = 16
    lr = 1e-3
    penalty_weight = 100
    num_data = 5000
    test_size = 1000
    val_size = 1000
    train_size = num_data - test_size - val_size

    # init model
    from src.problem import msRosenbrock
    model = msRosenbrock(steepness, num_blocks, timelimit=1000)

    # parameters as input data
    b_low, b_high = 1.0, 8.0
    a_low, a_high = 0.5, 4.5
    b_train = np.random.uniform(b_low, b_high, (train_size, 1)).astype(np.float32)
    b_test  = np.random.uniform(b_low, b_high, (test_size, 1)).astype(np.float32)
    b_val   = np.random.uniform(b_low, b_high, (val_size, 1)).astype(np.float32)
    a_train = np.random.uniform(a_low, a_high, (train_size, num_blocks)).astype(np.float32)
    a_test  = np.random.uniform(a_low, a_high, (test_size, num_blocks)).astype(np.float32)
    a_val   = np.random.uniform(a_low, a_high, (val_size, num_blocks)).astype(np.float32)
    # nm datasets
    from neuromancer.dataset import DictDataset
    data_train = DictDataset({"b":b_train, "a":a_train}, name="train")
    data_test = DictDataset({"b":b_test, "a":a_test}, name="test")
    data_val = DictDataset({"b":b_val, "a":a_val}, name="dev")
    # torch dataloaders
    from torch.utils.data import DataLoader
    batch_size = 64
    loader_train = DataLoader(data_train, batch_size, num_workers=0, collate_fn=data_train.collate_fn, shuffle=True)
    loader_test = DataLoader(data_test, batch_size, num_workers=0, collate_fn=data_test.collate_fn, shuffle=False)
    loader_val = DataLoader(data_val, batch_size, num_workers=0, collate_fn=data_val.collate_fn, shuffle=True)

    # define neural architecture for the solution map smap(p, a) -> x
    import neuromancer as nm
    func = nm.modules.blocks.MLP(insize=num_blocks+1, outsize=2*num_blocks-3, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["b", "a"], ["z"], name="smap")

    # linear constraint encode
    encoding = equalityEncoding(num_blocks, input_key="z", output_key="x")

    # define rounding model
    from src.func.layer import netFC
    from src.func import roundGumbelModel
    layers_rnd = netFC(input_dim=3*num_blocks+1, hidden_dims=[hsize]*hlayers_rnd, output_dim=2*num_blocks-3)
    rnd = roundGumbelModel(layers=layers_rnd, param_keys=["b", "a"], var_keys=["x"], output_keys=["x_rnd"],
                           int_ind=model.int_ind, continuous_update=True, equality_encoding=encoding, name="round")

    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, encoding, rnd])
    loss_fn = penaltyLoss(["b", "a", "x_rnd"], steepness, num_blocks, penalty_weight)

    # training
    from src.problem.neuromancer.trainer import trainer
    epochs = 200                    # number of training epochs
    patience = 20                   # number of epochs with no improvement in eval metric to allow before early stopping
    growth_rate = 1             # growth rate of penalty weight
    warmup = 20                 # number of epochs to wait before enacting early stopping policies
    optimizer = torch.optim.AdamW(components.parameters(), lr=lr)
    # create a trainer for the problem
    my_trainer = trainer(components, loss_fn, optimizer, epochs=epochs,
                         patience=patience, warmup=warmup)
    # training for the rounding problem
    my_trainer.train(loader_train, loader_val)

    # test neuroMANCER
    from src.utlis import nm_test_solve
    p, a = data_train[0]["b"].tolist(), data_train[0]["a"].tolist()
    datapoint = {"b": torch.tensor([p], dtype=torch.float32),
                 "a": torch.tensor([a], dtype=torch.float32),
                 "name":"test"}
    model.set_param_val({"b":p, "a":a})
    print("neuroMANCER:")
    nm_test_solve("x", components, datapoint, model)
