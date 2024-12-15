"""
Parametric Simple Nonconvex Problem
"""

import numpy as np
from scipy.linalg import null_space
import torch
from torch import nn
import neuromancer as nm

from src.utlis import identityTransform

class penaltyLoss(nn.Module):
    """
    Penalty loss function for simple nonconvex problem
    """
    def __init__(self, input_keys, num_var, num_eq, num_ineq, penalty_weight=50, output_key="loss"):
        super().__init__()
        self.b_key, self.x_key = input_keys
        self.output_key = output_key
        self.penalty_weight = penalty_weight
        self.device = None
        # fixed coefficients
        rng = np.random.RandomState(17)
        Q = 0.01 * np.diag(rng.random(size=num_var))
        p = 0.1 * rng.random(num_var)
        G = rng.normal(scale=0.1, size=(num_eq, num_var))
        A = rng.normal(scale=0.1, size=(num_ineq, num_var))
        self.Q = torch.from_numpy(Q).float()
        self.p = torch.from_numpy(p).float()
        self.G = torch.from_numpy(G).float()
        self.A = torch.from_numpy(A).float()

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
        x = input_dict[self.x_key]
        # update device
        if self.device is None:
            self.device = x.device
            self.Q = self.Q.to(self.device)
            self.p = self.p.to(self.device)
            self.A = self.A.to(self.device)
        # 1/2 x^T Q x
        Q_term = torch.einsum("bm,nm,bm->b", x, self.Q, x) / 2
        # p^T y
        p_term = torch.einsum("m,bm->b", self.p, torch.sin(x))
        return Q_term + p_term

    def cal_constr_viol(self, input_dict):
        """
        calculate constraints violation
        """
        # get values
        x, b = input_dict[self.x_key], input_dict[self.b_key]
        # eq constraints
        lhs = torch.einsum("mn,bn->bm", self.G, x) # Gx
        rhs = 0
        eq_violation = (torch.abs(lhs - rhs)).sum(dim=1) # Ax<=b
        # ineq constraints
        lhs = torch.einsum("mn,bn->bm", self.A, x) # Ax
        rhs = b # b
        ineq_violation = (torch.relu(lhs - rhs)).sum(dim=1) # Ax<=b
        return eq_violation + ineq_violation


class equalityEncoding(nn.Module):
    def __init__(self, num_var, num_eq, input_key, output_key):
        """
        encode equality constraints G x = 0 using null space decomposition.
        """
        super().__init__()
        # size
        self.num_var = num_var
        self.num_eq = num_eq
        # data keys
        self.input_key = input_key
        self.output_key = output_key
        # init encoding
        rng = np.random.RandomState(17)
        Q = 0.01 * np.diag(rng.random(size=num_var))
        p = 0.1 * rng.random(num_var)
        G = rng.normal(scale=0.1, size=(num_eq, num_var))
        # sepecial solution for equality constraints
        x_s = np.zeros(num_var)
        # null space for equality constraints
        N = null_space(G)
        # reconstruct matrix
        N = identityTransform(N, num_var//2)
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
        x = torch.zeros((batch_size, self.num_var)).to(self.device)
        # continous part  to encode
        x = self.x_s + torch.einsum("bj,ij->bi", z, self.N)
        data[self.output_key] = x
        return data


if __name__ == "__main__":

    # random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # init
    num_var = 100
    num_eq = 10
    num_ineq = 90
    hlayers_sol = 5
    hlayers_rnd = 4
    hsize = 128
    lr = 1e-3
    penalty_weight = 100
    num_data = 10000
    test_size = 1000
    val_size = 1000
    train_size = num_data - test_size - val_size

    # init mathmatic model
    from src.problem.math_solver.nonconvex import nonconvex
    model = nonconvex(num_var, num_eq, num_ineq, timelimit=60)

    # data sample from uniform distribution
    b_samples = torch.from_numpy(np.random.uniform(-1, 1, size=(num_data, num_ineq))).float()
    data = {"b":b_samples}
    # data split
    from src.utlis import data_split
    data_train, data_test, data_val = data_split(data, test_size=test_size, val_size=val_size)
    # torch dataloaders
    from torch.utils.data import DataLoader
    batch_size = 64
    loader_train = DataLoader(data_train, batch_size, num_workers=0,
                              collate_fn=data_train.collate_fn, shuffle=True)
    loader_test  = DataLoader(data_test, batch_size, num_workers=0,
                              collate_fn=data_test.collate_fn, shuffle=False)
    loader_val   = DataLoader(data_val, batch_size, num_workers=0,
                              collate_fn=data_val.collate_fn, shuffle=False)

    # define neural architecture for the solution map smap(p) -> x
    import neuromancer as nm
    func = nm.modules.blocks.MLP(insize=num_ineq, outsize=num_var-num_eq, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[10]*4)
    smap = nm.system.Node(func, ["b"], ["z"], name="smap")

    # define rounding model
    int_ind = {"z":range(num_var//2)}
    from src.func.layer import netFC
    from src.func import roundGumbelModel
    layers_rnd = netFC(input_dim=num_ineq+num_var-num_eq,
                       hidden_dims=[hsize]*hlayers_rnd,
                       output_dim=num_var-num_eq)
    rnd = roundGumbelModel(layers=layers_rnd, param_keys=["b"], var_keys=["z"],
                           output_keys=["z_rnd"], int_ind=int_ind,
                           continuous_update=True, name="round")

    # linear constraint encode
    encoding = equalityEncoding(num_var, num_eq, input_key="z_rnd", output_key="x")

    # build neuromancer components
    components = nn.ModuleList([smap, rnd, encoding])

    # build neuromancer problem
    loss_fn = penaltyLoss(["b", "x"], num_var, num_eq, num_ineq)

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
    print("======================================================")
    print("neuroMANCER:")
    datapoint = {"b": b_samples[:1],
                 "name":"test"}
    model.set_param_val({"b":b_samples[0].cpu().numpy()})
    nm_test_solve("x", components, datapoint, model)

    # solve the MIQP
    from src.utlis import ms_test_solve
    print("======================================================")
    print("Gurobi:")
    ms_test_solve(model, tee=True)
