"""
Parametric Mixed Integer Rosenbrock Problem
"""

import numpy as np
import neuromancer as nm

from src.problem.neuromancer import abcNMProblem

class rosenbrock(abcNMProblem):
    def __init__(self, num_vars, vars, params, components, penalty_weight):
        # init
        self.num_vars = num_vars
        super().__init__(vars=vars, params=params, components=components, penalty_weight=penalty_weight)

    def getObj(self, vars, params):
        """
        Get neuroMANCER objective component
        """
        # get decision variables
        x, = vars.values()
        # get mutable parameters
        p, a = params.values()
        # objective function sum (1 - x_i)^2 + a (x_i+1 - x_i^2)^2
        f = sum((1 - x[:, i]) ** 2 + a[:, i] * (x[:, i + 1] - x[:, i] ** 2) ** 2
                 for i in range(self.num_vars-1))
        obj = f.minimize(weight=1.0, name="obj")
        return obj

    def getConstrs(self, vars, params, penalty_weight):
        """
        Get neuroMANCER constraint component
        """
        # get decision variables
        x, = vars.values()
        # get mutable parameters
        p, a = params.values()
        # constraints
        constraints = []
        # constraints 0:
        g = sum((-1) ** i * x[:, i] for i in range(self.num_vars))
        con = penalty_weight * (g >= 0)
        con.name = "c0"
        constraints.append(con)
        # constraints 1:
        g = sum(x[:, i] ** 2 for i in range(self.num_vars))
        con = penalty_weight * (g >= p[:, 0] / 2)
        con.name = "c1"
        constraints.append(con)
        # constraints 2:
        g = sum(x[:, i] ** 2 for i in range(self.num_vars))
        con = penalty_weight * (g <= p[:, 0])
        con.name = "c2"
        constraints.append(con)
        return constraints

if __name__ == "__main__":

    import torch
    from torch import nn

    # random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # init
    num_vars = 10     # number of decision variables
    num_data = 5000   # number of data
    test_size = 1000  # number of test size
    val_size = 1000   # number of validation size

    # data sample from uniform distribution
    p_low, p_high = 0.5, 6.0
    a_low, a_high = 0.2, 1.2
    p_samples = torch.FloatTensor(num_data, 1).uniform_(p_low, p_high)
    a_samples = torch.FloatTensor(num_data, num_vars-1).uniform_(a_low, a_high)
    data = {"p":p_samples, "a":a_samples}
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

    # define neural architecture for the solution map smap(p, a) -> x
    import neuromancer as nm
    func = nm.modules.blocks.MLP(insize=num_vars, outsize=num_vars, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[10]*4)
    components = [nm.system.Node(func, ["p", "a"], ["x"], name="smap")]
    # build neuromancer problem
    problem = rosenbrock(num_vars=num_vars, vars=["x"], params=["p", "a"],
                         components=components, penalty_weight=100)

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
    from src.problem.math_solver.rosenbrock import rosenbrock
    model = rosenbrock(num_vars=num_vars)

    # test neuroMANCER
    from src.utlis import nmSolveTest
    print("neuroMANCER:")
    datapoint = {"p": torch.tensor([[1.2]], dtype=torch.float32),
                 "a": torch.tensor([list(np.random.uniform(0.2, 1.2, num_vars-1))], dtype=torch.float32),
                 "name":"test"}
    nmSolveTest(problem, datapoint, model)
