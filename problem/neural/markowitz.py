import neuromancer as nm

def nnMarkowitz(exp_returns, cov_matrix, func, alpha=100):
    # solution map from model parameters: sol_map(p) -> x
    sol_map = nm.system.Node(func, ["p"], ["x"], name="smap")
    # trainable components
    components = [sol_map]
    # parameters
    p = nm.constraint.variable("p")
    # variables
    x = nm.constraint.variable("x")
    # obj & constr
    obj, constrs = probMarkowitz(x, p, exp_returns, cov_matrix, alpha)
    # merit loss function
    loss = nm.loss.PenaltyLoss(obj, constrs)
    # optimization solver
    problem = nm.problem.Problem(components, loss)
    return problem


def probMarkowitz(x, p, exp_returns, cov_matrix, alpha=100):
    # number of vars
    num_vars = len(exp_returns)
    # objective function
    f = sum(cov_matrix[i,j] * x[:, i] * x[:, j] for i in range(num_vars) for j in range(num_vars))
    obj = f.minimize(weight=1.0, name="obj")
    objectives = [obj]
    # constraints
    constraints = []
    # constr: 100 units
    con = alpha * (sum(x[:, i] for i in range(num_vars)) == 100)
    con.name = "c_units"
    constraints.append(con)
    # constr: expected return
    con = alpha * (sum(exp_returns[i] * x[:, i] for i in range(num_vars)) >= 100 * p[:, 0])
    con.name = "c_return"
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
    num_data = 1000   # number of data
    num_vars = 10     # number of decision variables
    test_size = 200   # number of test size
    val_size = 200    # number of validation size

    # get datasets
    from data import getDatasetMarkowitz
    exp_returns, cov_matrix, (data_train, data_test, data_dev) = getDatasetMarkowitz(num_data=num_data,
                                                                                     num_vars=num_vars,
                                                                                     test_size=test_size,
                                                                                     val_size=val_size)
    # expected returns
    print("Expected Returns:")
    print(exp_returns)
    # covariance matrix
    print("Covariance Matrix:")
    print(cov_matrix)

    # torch dataloaders
    from torch.utils.data import DataLoader
    loader_train = DataLoader(data_train, batch_size=32, num_workers=0, collate_fn=data_train.collate_fn, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=32, num_workers=0, collate_fn=data_test.collate_fn, shuffle=True)
    loader_dev = DataLoader(data_dev, batch_size=32, num_workers=0, collate_fn=data_dev.collate_fn, shuffle=True)

    # define neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=1, outsize=num_vars, bias=True,
                                 linear_map=nm.slim.maps["linear"], nonlin=nn.ReLU, hsizes=[5]*4)

    # get solver
    problem = nnMarkowitz(exp_returns, cov_matrix, func, alpha=100)

    # training
    print(0)
    lr = 0.001    # step size for gradient descent
    epochs = 20   # number of training epochs
    warmup = 50   # number of epochs to wait before enacting early stopping policy
    patience = 50 # number of epochs with no improvement in eval metric to allow before early stopping
    # set adamW as optimizer
    optimizer = torch.optim.AdamW(problem.parameters(), lr=lr)

    # test
    for data in loader_train:
        # calculate loss
        loss = problem(data)["train_loss"]
        print(loss)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # define trainer
    print(1)
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
    print(2)
    best_model = trainer.train()
    print()