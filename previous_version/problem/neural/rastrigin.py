import math

import neuromancer as nm

def nnRastrigin(num_vars, func, alpha=100):
    # features from parameters: z := <p, a>
    feat = nm.system.Node(lambda p, a: torch.cat([p, a], dim=-1), ["p", "a"], ["z"], name="feat")
    # solution map from model parameters: sol_map(z) -> x
    sol_map = nm.system.Node(func, ["z"], ["x"], name="smap")
    # trainable components
    components = [feat, sol_map]
    # parameters
    p = nm.constraint.variable("p")
    a = nm.constraint.variable("a")
    # variables
    x = nm.constraint.variable("x")
    # penalty loss
    loss = lossRastrigin(x, p, a, num_vars, alpha)
    # optimization solver
    problem = nm.problem.Problem(components, loss)
    return problem


def lossRastrigin(x, p, a, num_vars, alpha=100):
    # objective function
    f = sum(a[:, i] + x[:, i] ** 2 - a[:, i] * torch.cos(2 * math.pi * x[:, i]) for i in range(num_vars))
    obj = f.minimize(weight=1.0, name="obj")
    objectives = [obj]
    # constraints
    constraints = []
    # constraints 0:
    g = sum((-1) ** i * x[:, i] for i in range(num_vars))
    con = alpha * (g >= 0)
    con.name = "c0"
    constraints.append(con)
    # constraints 1:
    g = sum(x[:, i] ** 2 for i in range(num_vars))
    con = alpha * (g >= p[:, 0] / 2)
    con.name = "c1"
    constraints.append(con)
    # constraints 2:
    g = sum(x[:, i] ** 2 for i in range(num_vars))
    con = alpha * (g <= p[:, 0])
    con.name = "c2"
    constraints.append(con)
    # variable bound
    for i in range(num_vars):
        lb = alpha * (x[:, i] >= -5.12)
        lb.name = "lb{}".format(i)
        constraints.append(lb)
        ub = alpha * (x[:, i] <= 5.12)
        ub.name = "ub{}".format(i)
        constraints.append(ub)
    # merit loss function
    loss = nm.loss.PenaltyLoss(objectives, constraints)
    return loss


if __name__ == "__main__":

    import numpy as np
    import torch
    from torch import nn

    from utlis import test

    # random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # init
    num_data = 5000   # number of data
    num_vars = 10     # number of decision variables
    num_ints = 5      # number of integer decision variables
    test_size = 1000  # number of test size
    val_size = 1000   # number of validation size

    # get datasets
    from data import getDatasetRatrigin
    data_train, data_test, data_dev = getDatasetRatrigin(num_data=num_data, num_vars=num_vars,
                                                         test_size=test_size, val_size=val_size)

    # torch dataloaders
    from torch.utils.data import DataLoader
    loader_train = DataLoader(data_train, batch_size=32, num_workers=0, collate_fn=data_train.collate_fn, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=32, num_workers=0, collate_fn=data_test.collate_fn, shuffle=True)
    loader_dev = DataLoader(data_dev, batch_size=32, num_workers=0, collate_fn=data_dev.collate_fn, shuffle=True)

    # define neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_vars+1, outsize=num_vars, bias=True,
                                 linear_map=nm.slim.maps["linear"], nonlin=nn.ReLU, hsizes=[50]*4)

    # get solver
    problem = nnRastrigin(num_vars, func, alpha=100)

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

    # params
    p = np.random.uniform(2, 6, 1)
    a = np.random.uniform(6, 15, num_vars)
    print("Parameters p:", list(p))
    print("Parameters a:", list(a))
    print()

    # get solution from Ipopt
    print("Ipopt:")
    from problem.solver import exactRastrigin
    model = exactRastrigin(n_vars=num_vars, n_integers=0)
    model.setParamValue(p, *a)
    test.solverTest(model, solver="ipopt")

    # get solution from neuroMANCER
    print("neuroMANCER:")
    datapoint = {"p": torch.tensor([list(p)], dtype=torch.float32),
                 "a": torch.tensor([list(a)], dtype=torch.float32),
                 "name":"test"}
    test.nmTest(problem, datapoint, model)
