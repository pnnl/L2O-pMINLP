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
        q = rng.normal(scale=1, size=(num_blocks))
        self.P = torch.from_numpy(P).float()
        self.q = torch.from_numpy(q).float()


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
            self.q = self.q.to(self.device)
        # inner constraint violation
        lhs_inner = torch.sum(x[:, 1::2], dim=1)
        rhs_inner = self.num_blocks * b[:, 0] / 2
        inner_violation = torch.relu(rhs_inner - lhs_inner)
        # outer constraint violation
        lhs_outer = torch.sum(x[:, ::2] ** 2, dim=1)
        rhs_outer = self.num_blocks * b[:, 0]
        outer_violation = torch.relu(lhs_outer - rhs_outer)
        # lear constraint violation
        lhs = torch.matmul(x[:, 1::2], self.q)
        linear_violation = torch.relu(lhs)
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
    steepness = 50    # steepness factor
    num_blocks = 10   # number of expression blocks
    num_data = 5000   # number of data
    test_size = 1000  # number of test size
    val_size = 1000   # number of validation size

    # data sample from uniform distribution
    b_low, b_high = 1.0, 8.0
    a_low, a_high = 0.5, 4.5
    b_samples = torch.FloatTensor(num_data, 1).uniform_(b_low, b_high)
    a_samples = torch.FloatTensor(num_data, num_blocks).uniform_(a_low, a_high)
    data = {"b":b_samples, "a":a_samples}
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

    # define neural architecture for the solution map smap(p, a) -> x
    import neuromancer as nm
    func = nm.modules.blocks.MLP(insize=num_blocks+1, outsize=2*num_blocks-3, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[64]*4)
    smap = nm.system.Node(func, ["b", "a"], ["z"], name="smap")

    # linear constraint encode
    encoding = equalityEncoding(num_blocks, input_key="z", output_key="x")

    # build neuromancer components
    components = nn.ModuleList([smap, encoding])

    #loss = PenaltyLoss(["p", "a", "x"], steepness, num_blocks)
    #for data_dict in loader_train:
        # add x to dict
    #    data_dict.update(components(data_dict))
        # calculate loss
    #    print(loss(data_dict))
    #    break

    # build neuromancer problems
    #loss = nm.loss.PenaltyLoss(obj, constrs)
    # problem = nm.problem.Problem([components], loss)
    loss_fn = penaltyLoss(["b", "a", "x"], steepness, num_blocks)

    # training
    lr = 0.0001    # step size for gradient descent
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
    from src.problem.math_solver.rosenbrock import rosenbrock
    model = rosenbrock(steepness=steepness, num_blocks=num_blocks)

    # test neuroMANCER
    from src.utlis import nm_test_solve
    p, a = data_train[0]["b"].tolist(), data_train[0]["a"].tolist()
    datapoint = {"b": torch.tensor([p], dtype=torch.float32),
                 "a": torch.tensor([a], dtype=torch.float32),
                 "name":"test"}
    model.set_param_val({"b":p, "a":a})
    print("neuroMANCER:")
    nm_test_solve("x", components, datapoint, model)
