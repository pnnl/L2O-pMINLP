"""
Parametric Mixed Integer Rosenbrock Problem
"""

import numpy as np
import neuromancer as nm

def rosenbrock(var_keys, param_keys, steepness, num_blocks, penalty_weight=100):
    # mutable parameters
    params = {}
    for p in param_keys:
        params[p] = nm.constraint.variable(p)
    # decision variables
    vars = {}
    for v in var_keys:
        vars[v] = nm.constraint.variable(v)
    obj = [get_obj(vars, params, steepness, num_blocks)]
    constrs = get_constrs(vars, params, num_blocks, penalty_weight)
    return obj, constrs


def get_obj(vars, params, steepness, num_blocks):
    """
    Get neuroMANCER objective component
    """
    # get decision variables
    x, = vars.values()
    # get mutable parameters
    p, a = params.values()
    # objective function sum (a_i - x_2i)^2 + b * (x_2i+1 - x_2i^2)^2
    f = sum((a[:, i] - x[:, 2*i]) ** 2 + steepness * (x[:, 2*i+1] - x[:, 2*i] ** 2) ** 2
             for i in range(num_blocks))
    obj = f.minimize(weight=1.0, name="obj")
    return obj


def get_constrs(vars, params, num_blocks, penalty_weight):
    """
    Get neuroMANCER constraint component
    """
    # get decision variables
    x, = vars.values()
    # get mutable parameters
    p, a = params.values()
    # constraints
    constraints = []
    # inner ball:
    g = sum(x[:, 2*i] ** 2 for i in range(num_blocks))
    con = penalty_weight * (g >= num_blocks * p[:, 0] / 2)
    con.name = "c_inner"
    constraints.append(con)
    # outer ball:
    g = sum(x[:, 2*i] ** 2 for i in range(num_blocks))
    con = penalty_weight * (g <= num_blocks * p[:, 0])
    con.name = "c_outer"
    constraints.append(con)
    return constraints


if __name__ == "__main__":

    import torch
    from torch import nn

    # random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # init
    steepness = 30    # steepness factor
    num_blocks = 3    # number of expression blocks
    num_data = 5000   # number of data
    test_size = 1000  # number of test size
    val_size = 1000   # number of validation size

    # data sample from uniform distribution
    p_low, p_high = 1.0, 8.0
    a_low, a_high = 0.5, 4.5
    p_samples = torch.FloatTensor(num_data, 1).uniform_(p_low, p_high)
    a_samples = torch.FloatTensor(num_data, num_blocks).uniform_(a_low, a_high)
    data = {"p":p_samples, "a":a_samples}
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
    obj, constrs = rosenbrock(["x"], ["p", "a"], steepness=steepness,
                              num_blocks=num_blocks, penalty_weight=100)

    # define neural architecture for the solution map smap(p, a) -> x
    import neuromancer as nm
    func = nm.modules.blocks.MLP(insize=num_blocks+1, outsize=2*num_blocks, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[10]*4)
    components = [nm.system.Node(func, ["p", "a"], ["x"], name="smap")]

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
    from src.problem.math_solver.rosenbrock import rosenbrock
    model = rosenbrock(steepness=steepness, num_blocks=num_blocks)

    # test neuroMANCER
    from src.utlis import nm_test_solve
    p, a = 3.2, list(np.random.uniform(a_low, a_high, num_blocks))
    datapoint = {"p": torch.tensor([[p]], dtype=torch.float32),
                 "a": torch.tensor([a], dtype=torch.float32),
                 "name":"test"}
    model.set_param_val({"p":p, "a":a})
    print("neuroMANCER:")
    nm_test_solve(["x"], problem, datapoint, model)
