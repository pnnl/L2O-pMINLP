"""
Differentiable/Learnable to project
"""
import torch
from torch import nn

from neuromancer.modules.solvers import GradientProjection as gradProj
from neuromancer.gradients import gradient

if __name__ == "__main__":

    import numpy as np

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

    # get objective function & constraints
    from src.problem import nmQuadratic
    obj_bar, constrs_bar = nmQuadratic(["x_bar"], ["p"], penalty_weight=100)
    obj_rnd, constrs_rnd = nmQuadratic(["x_rnd"], ["p"], penalty_weight=100)

    # define neural architecture for the solution map
    import neuromancer as nm
    func = nm.modules.blocks.MLP(insize=2, outsize=4, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[10]*4)
    smap = nm.system.Node(func, ["p"], ["x"], name="smap")

    # define rounding model
    from src.func.layer import netFC
    layers_rnd = netFC(input_dim=6, hidden_dims=[20]*3, output_dim=4)
    from src.func.rnd import roundModel, roundGumbelModel, roundThresholdModel
    #round_func = roundModel(layers=layers_rnd, param_keys=["p"], var_keys=["x"], output_keys=["x_rnd"],
    #                        int_ind={"x":[2,3]}, continuous_update=False, name="round")
    round_func = roundGumbelModel(layers=layers_rnd, param_keys=["p"], var_keys=["x"], output_keys=["x_rnd"],
                                  bin_ind={"x":[2,3]}, continuous_update=False, name="round")
    #round_func = roundThresholdModel(layers=layers_rnd, param_keys=["p"], var_keys=["x"], output_keys=["x_rnd"],,
    #                                 int_ind={"x":[2,3]}, continuous_update=False, name="round")

    # define projection layer
    num_steps = 10
    step_size = 0.1
    decay = 0.1
    proj = gradProj(constraints=constrs_rnd, input_keys=["x_rnd"], output_keys=["x_bar"],
                    num_steps=num_steps, step_size=step_size, decay=decay)

    # build neuromancer problem
    components = [smap, round_func, proj]
    loss = nm.loss.PenaltyLoss(obj_bar, constrs_bar)
    problem = nm.problem.Problem(components, loss, grad_inference=True)

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
    nmSolveTest(["x"], problem, datapoint, model)
    nmSolveTest(["x_rnd"], problem, datapoint, model)
    nmSolveTest(["x_bar"], problem, datapoint, model)
