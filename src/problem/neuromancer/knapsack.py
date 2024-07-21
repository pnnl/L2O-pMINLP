"""
Parametric Multi-Dimensional Knapsack
"""

import numpy as np
import neuromancer as nm


def knapsack(var_keys, param_keys, weights, caps, penalty_weight=100):
    # mutable parameters
    params = {}
    for p in param_keys:
        params[p] = nm.constraint.variable(p)
    # decision variables
    vars = {}
    for v in var_keys:
        vars[v] = nm.constraint.variable(v)
    obj = [get_obj(vars, params)]
    constrs = get_constrs(vars, weights, caps, penalty_weight)
    return obj, constrs


def get_obj(vars, params):
    """
    Get neuroMANCER objective component
    """
    # get decision variables
    x, = vars.values()
    # get mutable parameters
    c = params.values()
    # objective function c^T x
    f = sum(- ci * xi for ci, xi in zip(c, x))
    obj = f.minimize(weight=1.0, name="obj")
    return obj


def get_constrs(vars, weights, caps, penalty_weight):
    """
    Get neuroMANCER constraint component
    """
    # problem size
    dim, num_var = weights.shape
    # get decision variables
    x, = vars.values()
    # constraints
    constraints = []
    for i in range(dim):
        g = sum(weights[i, j] * x[:, j] for j in range(num_var))
        con = penalty_weight * (g <= caps[i])
        con.name = "cap_{}".format(i)
        constraints.append(con)
    return constraints


if __name__ == "__main__":

    import torch
    from torch import nn

    # random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # init
    num_var = 32      # number of variables
    #num_var = 15      # number of variables
    dim = 2           # dimension of constraints
    caps = [20] * dim # capacity
    num_data = 5000   # number of data
    test_size = 1000  # number of test size
    val_size = 1000   # number of validation size

    # data sample from PyEPO
    import pyepo
    # generate data
    weights, x, c = pyepo.data.knapsack.genData(num_data=num_data, num_features=5,
                                                num_items=num_var, dim=dim,
                                                deg=4, noise_width=0.5)
    c_samples = torch.FloatTensor(c)
    data = {"c":c_samples}
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
    obj, constrs = knapsack(["x"], ["c"], weights=weights, caps=caps, penalty_weight=100)


    # define neural architecture for the solution map smap(c) -> x
    import neuromancer as nm
    func = nm.modules.blocks.MLP(insize=num_var, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[64]*2)
    components = [nm.system.Node(func, ["c"], ["x"], name="smap")]

    # build neuromancer problems
    loss = nm.loss.PenaltyLoss(obj, constrs)
    problem = nm.problem.Problem(components, loss)

    # training
    lr = 0.001    # step size for gradient descent
    epochs = 400  # number of training epochs
    warmup = 40   # number of epochs to wait before enacting early stopping policy
    patience = 40 # number of epochs with no improvement in eval metric to allow before early stopping
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
    # load best model dict
    problem.load_state_dict(best_model)
    print()

    # init mathmatic model
    from src.problem.math_solver.kanpsack import kanpsack
    model = kanpsack(weights=weights, caps=caps)

    # test neuroMANCER
    from src.utlis import nm_test_solve
    c = c[0]
    datapoint = {"c": torch.tensor([c], dtype=torch.float32),
                 "name":"test"}
    model.set_param_val({"c":c})
    print("neuroMANCER:")
    nm_test_solve(["x"], problem, datapoint, model)
