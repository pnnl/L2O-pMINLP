"""
Parametric Mixed Integer Quadratic Programming

https://www.sciencedirect.com/science/article/pii/S0098135401007979
https://www.sciencedirect.com/science/article/pii/S1570794601801577
"""

import numpy as np
import neuromancer as nm

from src.problem.neuromancer import abcNMProblem

class quadratic(abcNMProblem):
    def __init__(self, vars, params, components, penalty_weight):
        # define the coefficients for the quadratic problem
        self.c = [0.0200,
                  0.0300]
        self.Q = [[0.0196, 0.0063],
                  [0.0063, 0.0199]]
        self.d = [-0.3000,
                  -0.3100]
        self.b = [0.417425,
                  3.582575,
                  0.413225,
                  0.467075,
                  1.090200,
                  2.909800,
                  1.000000]
        self.A = [[ 1.0000,  0.0000],
                  [-1.0000,  0.0000],
                  [-0.0609,  0.0000],
                  [-0.0064,  0.0000],
                  [ 0.0000,  1.0000],
                  [ 0.0000, -1.0000],
                  [ 0.0000,  0.0000]]
        self.E = [[-1.0000,  0.0000],
                  [-1.0000,  0.0000],
                  [ 0.0000, -0.5000],
                  [ 0.0000, -0.7000],
                  [-0.6000,  0.0000],
                  [-0.5000,  0.0000],
                  [ 1.0000,  1.0000]]
        self.F = [[ 3.16515,  3.7546],
                  [-3.16515, -3.7546],
                  [ 0.17355, -0.2717],
                  [ 0.06585,  0.4714],
                  [ 1.81960, -3.2841],
                  [-1.81960,  3.2841],
                  [ 0.00000,  0.0000]]
        # init
        super().__init__(vars=vars, params=params,
                         components=components, penalty_weight=penalty_weight)

    def getObj(self, vars, params):
        """
        Get neuroMANCER objective component
        """
        # get decision variables
        x, = vars.values()
        # objective function C^T x + 1/2 x^T Q x + d^T y
        f = sum(self.c[j] * x[:, j] for j in range(2)) \
          + 0.5 * sum(self.Q[i][j] * x[:, i] * x[:, j] for i in range(2) for j in range(2)) \
          + sum(self.d[j] * x[:, j+2] for j in range(2))
        obj = f.minimize(weight=1.0, name="obj")
        return obj

    def getConstrs(self, vars, params, penalty_weight):
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
            lhs = sum(self.A[i][j] * x[:, j] for j in range(2)) \
                + sum(self.E[i][j] * x[:, j+2] for j in range(2))
            rhs = self.b[i] + self.F[i][0] * p[:, 0] + self.F[i][1] * p[:, 1]
            con = penalty_weight * (lhs <= rhs)
            con.name = "c{}".format(i)
            constraints.append(con)
        # nonnegative
        for i in range(4):
            con = penalty_weight * (x[:, i] >= 0)
            con.name = "xl{}".format(i)
            constraints.append(con)
        return constraints

if __name__ == "__main__":

    import torch
    from torch import nn

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

    # define neural architecture for the solution map smap(p) -> x
    import neuromancer as nm
    func = nm.modules.blocks.MLP(insize=2, outsize=4, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[10]*4)
    components = [nm.system.Node(func, ["p"], ["x"], name="smap")]
    # build neuromancer problem
    problem = quadratic(vars=["x"], params=["p"], components=components, penalty_weight=100)

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
    nmSolveTest(problem, datapoint, model)
