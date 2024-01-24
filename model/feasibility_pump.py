import neuromancer as nm
from neuromancer.modules.solvers import GradientProjection as gradProj

from model.round import roundModel

def feasibilityPumpModel(input_keys, getProb, sol_func, rnd_layer, int_ind, num_iters):
    # init trainable components
    components = []
    # parameters
    params = []
    for key in input_keys:
        params.append(nm.constraint.variable(key))
    # solution map from model parameters: sol_map(p) -> x
    sol_map = nm.system.Node(sol_func, input_keys, ["x_bar_0"], name="smap") # mapping
    components.append(sol_map)
    x_bar = nm.constraint.variable("x_bar_0") # variables
    obj_bar, constrs_bar = getProb(x_bar, *params) # obj & constr
    loss = nm.loss.PenaltyLoss(obj_bar, constrs_bar) # penalty loss
    # get problem
    problem_rel = nm.problem.Problem([sol_map], loss)
    # feasibility pump iterations
    for i in range(num_iters):
        # rounding step
        x_rnd = nm.constraint.variable("x_rnd_{}".format(i)) # variables
        obj_rnd, constrs_rnd = getProb(x_rnd, *params) # obj & constr
        loss += nm.loss.PenaltyLoss(obj_rnd, constrs_rnd) # penalty loss
        round_func = roundModel(layers=rnd_layer, param_keys=input_keys,
                                var_keys=["x_bar_{}".format(i)],
                                output_keys=["x_rnd_{}".format(i)],
                                int_ind={"x_bar_{}".format(i):int_ind},
                                name="round_{}".format(i))
        components.append(round_func)
        # projection step
        num_steps = 10
        step_size = 0.1
        decay = 0.1
        grad_proj = gradProj(constraints=constrs_rnd,
                             input_keys=["x_rnd_{}".format(i)],
                             output_keys=["x_bar_{}".format(i+1)],
                             num_steps=num_steps, step_size=step_size, decay=decay,
                             name="proj_{}".format(i))
        components.append(grad_proj)
    # last step: rounding
    x_rnd = nm.constraint.variable("x_rnd") # variables
    obj_rnd, constrs_rnd = getProb(x_rnd, *params) # obj & constr
    loss += nm.loss.PenaltyLoss(obj_rnd, constrs_rnd) # penalty loss
    #losses += loss
    round_func = roundModel(layers=rnd_layer, param_keys=input_keys,
                            var_keys=["x_bar_{}".format(i+1)],
                            output_keys=["x_rnd"],
                            int_ind={"x_bar_{}".format(i+1):int_ind},
                            name="round")
    components.append(round_func)
    # get problem
    problem_fp = nm.problem.Problem(components, loss, grad_inference=True)
    return problem_rel, problem_fp


if __name__ == "__main__":

    # add system path
    import sys
    import os
    sys.path.append(os.path.abspath("."))
    sys.path.append(os.path.abspath(".."))

    import numpy as np
    import torch
    from torch import nn

    from problem.solver import exactQuadratic
    from model.layer import netFC

    # random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # init
    num_data = 5000   # number of data
    num_vars = 5      # number of decision variables
    num_ints = 5      # number of integer decision variables
    test_size = 1000  # number of test size
    val_size = 1000   # number of validation size

    # exact optimization model
    model = exactQuadratic(n_vars=num_vars, n_integers=num_ints)

    # get datasets
    from data import getDatasetQradratic
    data_train, data_test, data_dev = getDatasetQradratic(num_data=num_data, num_vars=num_vars,
                                                          test_size=test_size, val_size=val_size)

    # torch dataloaders
    from torch.utils.data import DataLoader
    loader_train = DataLoader(data_train, batch_size=32, num_workers=0, collate_fn=data_train.collate_fn, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=32, num_workers=0, collate_fn=data_test.collate_fn, shuffle=True)
    loader_dev = DataLoader(data_dev, batch_size=32, num_workers=0, collate_fn=data_dev.collate_fn, shuffle=True)

    # define neural architecture for the solution mapping
    sol_func = nm.modules.blocks.MLP(insize=num_vars, outsize=num_vars, bias=True,
                                     linear_map=nm.slim.maps["linear"], nonlin=nn.ReLU,
                                     hsizes=[80]*4)
    # define neural architecture for rounding
    rnd_layer = netFC(input_dim=num_vars*2, hidden_dims=[80]*4, output_dim=num_vars)

    # function to get nm optimization model
    from problem.neural import probQuadratic
    def getProb(x, p):
        return probQuadratic(x, p, num_vars=num_vars, alpha=100)

    # parameters
    p = nm.constraint.variable("p")

    # feasibility pump model
    feasibilityPumpModel(["p", "a"], getProb, sol_func, rnd_layer, int_ind=model.intInd, num_iters=10)
