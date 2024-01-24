import neuromancer as nm

def nnQuadratic(num_vars, func, alpha=100):
    # solution map from model parameters: sol_map(p) -> x
    sol_map = nm.system.Node(func, ["p"], ["x"], name="smap")
    # trainable components
    components = [sol_map]
    # parameters
    p = nm.constraint.variable("p")
    # variables
    x = nm.constraint.variable("x")
    # obj & constr
    obj, constrs = probQuadratic(x, p, num_vars, alpha)
    # merit loss function
    loss = nm.loss.PenaltyLoss(obj, constrs)
    # optimization solver
    problem = nm.problem.Problem(components, loss)
    return problem


def probQuadratic(x, p, num_vars, alpha=100):
    # name as uid
    name = x.key
    # objective function
    f = sum(x[:, i] ** 2 for i in range(num_vars))
    obj = f.minimize(weight=1.0, name="obj_"+name)
    objectives = [obj]
    # constraints
    constraints = []
    for i in range(num_vars - 1):
        sign = (-1) ** i
        g = x[:, i] + sign * x[:, i + 1] - p[:, i]
        # x[i] + x[i+1] - p[i] >= 0
        con = alpha * (g >= 0)
        con.name = "c{}_l_".format(i) + name
        constraints.append(con)
        # x[i] + x[i+1] - p[i] <= 5
        con = alpha * (g <= 5)
        con.name = "c{}_u_".format(i) + name
        constraints.append(con)
    g = x[:, -1] - x[:, 0] - p[:, -1]
    # x[-1] - x[0] - p[-1] >= 0
    con = alpha * (g >= 0)
    con.name = "c{}_l_".format(num_vars - 1) + name
    constraints.append(con)
    # x[-1] - x[0] - p[-1] <= 5
    con = alpha * (g <= 5)
    con.name = "c{}_u_".format(num_vars - 1) + name
    constraints.append(con)
    return objectives, constraints


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
    num_vars = 5      # number of decision variables
    num_ints = 5      # number of integer decision variables
    test_size = 1000  # number of test size
    val_size = 1000   # number of validation size

    # get datasets
    from data import getDatasetQradratic
    data_train, data_test, data_dev = getDatasetQradratic(num_data=num_data, num_vars=num_vars,
                                                          test_size=test_size, val_size=val_size)

    # torch dataloaders
    from torch.utils.data import DataLoader
    loader_train = DataLoader(data_train, batch_size=32, num_workers=0, collate_fn=data_train.collate_fn, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=32, num_workers=0, collate_fn=data_test.collate_fn, shuffle=True)
    loader_dev = DataLoader(data_dev, batch_size=32, num_workers=0, collate_fn=data_dev.collate_fn, shuffle=True)

    # define neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_vars, outsize=num_vars, bias=True,
                                 linear_map=nm.slim.maps["linear"], nonlin=nn.ReLU, hsizes=[50]*4)

    # get solver
    problem = nnQuadratic(num_vars, func, alpha=100)

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
    p = np.random.uniform(1, 11, num_vars)
    print("Parameters p:", list(p))
    print()

    # get solution from Ipopt
    print("Ipopt:")
    from problem.solver import exactQuadratic
    model = exactQuadratic(n_vars=num_vars, n_integers=0)
    model.setParamValue(*p)
    test.solverTest(model, solver="ipopt")

    # get solution from neuroMANCER
    print("neuroMANCER:")
    datapoint = {"p": torch.tensor([list(p)], dtype=torch.float32),
                 "name":"test"}
    test.nmTest(problem, datapoint, model)
