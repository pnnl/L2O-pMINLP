"""
Parametric Mixed Integer Quadratic Programming

https://www.sciencedirect.com/science/article/pii/S0098135401007979
https://www.sciencedirect.com/science/article/pii/S1570794601801577
"""

import numpy as np
import torch
from torch import nn
import neuromancer as nm

def quadratic(var_keys, param_keys, penalty_weight=10):
    # define the coefficients for the quadratic problem
    c = [0.0200,
         0.0300]
    Q = [[0.0196, 0.0063],
         [0.0063, 0.0199]]
    d = [-0.3000,
         -0.3100]
    b = [0.417425,
         3.582575,
         0.413225,
         0.467075,
         1.090200,
         2.909800,
         1.000000]
    A = [[ 1.0000,  0.0000],
         [-1.0000,  0.0000],
         [-0.0609,  0.0000],
         [-0.0064,  0.0000],
         [ 0.0000,  1.0000],
         [ 0.0000, -1.0000],
         [ 0.0000,  0.0000]]
    E = [[-1.0000,  0.0000],
         [-1.0000,  0.0000],
         [ 0.0000, -0.5000],
         [ 0.0000, -0.7000],
         [-0.6000,  0.0000],
         [-0.5000,  0.0000],
         [ 1.0000,  1.0000]]
    F = [[ 3.16515,  3.7546],
         [-3.16515, -3.7546],
         [ 0.17355, -0.2717],
         [ 0.06585,  0.4714],
         [ 1.81960, -3.2841],
         [-1.81960,  3.2841],
         [ 0.00000,  0.0000]]
    # mutable parameters
    params = {}
    for p in param_keys:
        params[p] = nm.constraint.variable(p)
    # decision variables
    vars = {}
    for v in var_keys:
        vars[v] = nm.constraint.variable(v)
    obj = [get_obj(vars, c, Q, d)]
    constrs = get_constrs(vars, params, penalty_weight, b, A, E, F)
    return obj, constrs


def get_obj(vars, c, Q, d):
    """
    Get neuroMANCER objective component
    """
    # get decision variables
    x, = vars.values()
    # objective function C^T x + 1/2 x^T Q x + d^T y
    f = sum(c[j] * x[:, j] for j in range(2)) \
      + 0.5 * sum(Q[i][j] * x[:, i] * x[:, j] for i in range(2) for j in range(2)) \
      + sum(d[j] * x[:, j+2] for j in range(2))
    obj = f.minimize(weight=1.0, name="obj")
    return obj

def get_constrs(vars, params, penalty_weight, b, A, E, F):
    """
    Get neuroMANCER constraint component
    """
    # get decision variables
    x, = vars.values()
    # get mutable parameters
    p, = params.values()
    # constraints
    constraints = []
    for i in range(7):
        lhs = sum(A[i][j] * x[:, j] for j in range(2)) \
            + sum(E[i][j] * x[:, j+2] for j in range(2))
        rhs = b[i] + F[i][0] * p[:, 0] + F[i][1] * p[:, 1]
        con = penalty_weight * (lhs <= rhs)
        con.name = "c{}".format(i)
        constraints.append(con)
    # nonnegative
    for i in range(4):
        con = penalty_weight * (x[:, i] >= 0)
        con.name = "xl{}".format(i)
        constraints.append(con)
    return constraints

class penaltyLoss(nn.Module):
    """
    Penalty loss function for quadratic problem
    """
    def __init__(self, input_keys, penalty_weight=50, output_key="loss"):
        super().__init__()
        self.p_key, self.x_key = input_keys
        self.output_key = output_key
        self.penalty_weight = penalty_weight
        # fixed coefficients
        self.c = torch.tensor([0.0200, 0.0300])
        self.Q = torch.tensor([[0.0196, 0.0063],
                              [0.0063, 0.0199]])
        self.d = torch.tensor([-0.3000, -0.3100])
        self.b = torch.tensor([0.417425, 3.582575, 0.413225, 0.467075, 1.090200, 2.909800, 1.000000])
        self.A = torch.tensor([[ 1.0000,  0.0000],
                               [-1.0000,  0.0000],
                               [-0.0609,  0.0000],
                               [-0.0064,  0.0000],
                               [ 0.0000,  1.0000],
                               [ 0.0000, -1.0000],
                               [ 0.0000,  0.0000]])
        self.E = torch.tensor([[-1.0000,  0.0000],
                               [-1.0000,  0.0000],
                               [ 0.0000, -0.5000],
                               [ 0.0000, -0.7000],
                               [-0.6000,  0.0000],
                               [-0.5000,  0.0000],
                               [ 1.0000,  1.0000]])
        self.F = torch.tensor([[ 3.16515,  3.7546],
                               [-3.16515, -3.7546],
                               [ 0.17355, -0.2717],
                               [ 0.06585,  0.4714],
                               [ 1.81960, -3.2841],
                               [-1.81960,  3.2841],
                               [ 0.00000,  0.0000]])

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
        # C^T x
        c_term = torch.einsum("i,bi->b", self.c, x[:, :2])
        # 1/2 x^T Q x
        Q_term = torch.einsum("bi,ij,bj->b", x[:, :2], self.Q, x[:, :2]) / 2
        # d^T y: dot product of d and x[:, 2:]
        d_term = torch.einsum("i,bi->b", self.d, x[:, 2:])
        return c_term + Q_term + d_term

    def cal_constr_viol(self, input_dict):
        """
        calculate constraints violation
        """
        # get values
        x, p = input_dict[self.x_key], input_dict[self.p_key]
        # constraints
        lhs = torch.einsum("ij,bj->bi", self.A, x[:, :2]) + torch.einsum("ij,bj->bi", self.E, x[:, 2:])
        rhs = self.b + torch.einsum("ij,bj->bi", self.F, p)
        violation = torch.relu(lhs - rhs).sum(dim=1)
        # bounds
        violation += torch.relu(-x).sum(dim=1)
        return violation


if __name__ == "__main__":

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

    # define neural architecture for the solution map smap(p) -> x
    import neuromancer as nm
    func = nm.modules.blocks.MLP(insize=2, outsize=4, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[10]*4)
    components = nn.ModuleList([nm.system.Node(func, ["p"], ["x"], name="smap")])

    # build neuromancer problem
    loss_fn = penaltyLoss(["p", "x"])

    # training
    lr = 0.001    # step size for gradient descent
    epochs = 200  # number of training epochs
    warmup = 20   # number of epochs to wait before enacting early stopping policy
    patience = 20 # number of epochs with no improvement in eval metric to allow before early stopping
    # set adamW as optimizer
    optimizer = torch.optim.AdamW(components.parameters(), lr=lr)
    # training
    from src.problem.neuromancer.trainer import trainer
    my_trainer = trainer(components, loss_fn, optimizer, epochs, patience, warmup)
    my_trainer.train(loader_train, loader_dev)
    print()

    # init mathmatic model
    from src.problem.math_solver.quadratic import quadratic
    model = quadratic()

    # test neuroMANCER
    p = 0.6, 0.8
    from src.utlis import nm_test_solve
    print("neuroMANCER:")
    datapoint = {"p": torch.tensor([[*p]], dtype=torch.float32),
                 "name":"test"}
    model.set_param_val({"p":p})
    nm_test_solve("x", components, datapoint, model)
