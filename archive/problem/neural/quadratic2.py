import neuromancer as nm

def nnQuadratic2(c, Q, d, b, A, E, F, func, alpha=100):
    # solution map from model parameters: sol_map(p) -> x
    sol_map = nm.system.Node(func, ["p"], ["x"], name="smap")
    # trainable components
    components = [sol_map]
    # parameters
    p = nm.constraint.variable("p")
    # variables
    x = nm.constraint.variable("x")
    # obj & constr
    obj, constrs = probQuadratic2(x, p, c, Q, d, b, A, E, F, alpha)
    # merit loss function
    loss = nm.loss.PenaltyLoss(obj, constrs)
    # optimization solver
    problem = nm.problem.Problem(components, loss)
    return problem


def probQuadratic2(x, p, c, Q, d, b, A, E, F, alpha=100):
    # objective function
    f = sum(c[i] * x[:, i] for i in range(2)) \
      + 0.5 * sum(Q[i, j] * x[:, i] * x[:, j] for i in range(2) for j in range(2)) \
      + sum(d[i] * x[:, i+2] for i in range(2))
    obj = f.minimize(weight=1.0, name="obj")
    objectives = [obj]
    # constraints
    constraints = []
    for i in range(7):
        lhs = sum(A[i, j] * x[:, j] for j in range(2)) + sum(E[i, j] * x[:, j+2] for j in range(2))
        rhs = b[i] + F[i, 0] * p[:, 0] + F[i, 1] * p[:, 1]
        con = alpha * (lhs <= rhs)
        con.name = "c{}".format(i)
        constraints.append(con)
    # nonnegative
    for i in range(4):
        con = alpha * (x[:, i] >= 0)
        con.name = "xl{}".format(i)
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
    num_vars = 10     # number of decision variables
    num_ints = 5      # number of integer decision variables
    test_size = 1000  # number of test size
    val_size = 1000   # number of validation size

    # get datasets
    from data import getDatasetQradratic2

    c, Q, d, b, A, E, F, datasets = getDatasetQradratic2(num_data=num_data, test_size=test_size, val_size=val_size)
    data_train, data_test, data_dev = datasets

    # torch dataloaders
    from torch.utils.data import DataLoader
    loader_train = DataLoader(data_train, batch_size=32, num_workers=0, collate_fn=data_train.collate_fn, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=32, num_workers=0, collate_fn=data_test.collate_fn, shuffle=True)
    loader_dev = DataLoader(data_dev, batch_size=32, num_workers=0, collate_fn=data_dev.collate_fn, shuffle=True)

    # define neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=2, outsize=4, bias=True,
                                 linear_map=nm.slim.maps["linear"], nonlin=nn.ReLU, hsizes=[10]*4)

    # get solver
    problem = nnQuadratic2(c, Q, d, b, A, E, F, func, alpha=100)

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
    p = np.random.uniform(0.5, 1, 2)
    print("Parameters p:", list(p))
    print()

    # get solution from Ipopt
    print("Ipopt:")
    from problem.solver import exactQuadratic2
    c, Q, d, b, A, E, F = c.cpu().numpy(), Q.cpu().numpy(), d.cpu().numpy(), b.cpu().numpy(), A.cpu().numpy(), E.cpu().numpy(), F.cpu().numpy()
    model = exactQuadratic2(c, Q, d, b, A, E, F).relax()
    model.setParamValue(*p)
    test.solverTest(model, solver="ipopt")

    # get solution from neuroMANCER
    print("neuroMANCER:")
    datapoint = {"p": torch.tensor([list(p)], dtype=torch.float32),
                 "name":"test"}
    test.nmTest(problem, datapoint, model)
