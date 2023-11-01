from torch import nn

from neuromancer.modules.solvers import GradientProjection as gradProj
from neuromancer.gradients import gradient

from model.layer import layerFC

class solPredGradProj(nn.Module):
    """
    re-predict solution then project onto feasible region
    """
    def __init__(self, constraints, input_dim, hidden_dims, output_dim,
                 param_keys, var_keys, output_keys=[],
                 residual=True, decay=0.1, num_steps=1, step_size=0.01, name=None):
        super(solPredGradProj, self).__init__()
        # data keys
        self.param_keys, self.var_keys, self.output_keys = param_keys, var_keys, output_keys
        self.input_keys = self.param_keys + self.var_keys
        self.output_keys = output_keys if output_keys else [self.var_key]
        # name
        self.name = name
        # list of neuromancer constraints
        self.constraints = constraints
        # flag to only predict residual of solution
        self.residual = residual
        # projection
        self.num_steps = num_steps
        self.step_size = step_size
        self.decay = decay
        self.gradProj = gradProj(constraints=self.constraints, input_keys=self.var_keys, output_keys=self.output_keys,
                                 num_steps=self.num_steps, step_size=self.step_size, decay=self.decay)
        # sequence
        self.layers = self._getSequence(input_dim, hidden_dims, output_dim)

    def forward(self, data):
        # get vars & params
        p, x = [data[k] for k in self.param_keys], [data[k] for k in self.var_keys]
        # get grad of violation
        energy = self.gradProj.con_viol_energy(data)
        grads = []
        for k in self.var_keys:
            step = gradient(energy, data[k]).detach()
            grads.append(step)
        # concatenate all features: params + sol
        f = torch.cat(p + x + grads, dim=-1)
        # forward
        out = self.layers(f)
        for k_in, k_out in zip(self.var_keys, self.output_keys):
            # new solution
            data[k_out] = self.residual * data[k_in] + out[:,:data[k_in].shape[1]+1]
            # cut off out
            out = out[:,data[k_in].shape[1]+1:]
        return data


    def _getSequence(self, input_dim, hidden_dims, output_dim):
        sizes = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(layerFC(sizes[i], sizes[i + 1]))
        return nn.Sequential(*layers)


if __name__ == "__main__":

    import numpy as np
    import torch
    from torch import nn
    import neuromancer as nm

    from problem.solver import exactQuadratic
    from model import roundModel, roundGumbelModel
    from heuristic import naive_round
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

    # parameters
    p = nm.constraint.variable("p")
    # variables
    x_bar = nm.constraint.variable("x_bar")
    x_rnd = nm.constraint.variable("x_rnd")
    # model
    from problem.neural import probQuadratic
    obj_bar, constrs_bar = probQuadratic(x_bar, p, num_vars=10, alpha=100)
    obj_rnd, constrs_rnd = probQuadratic(x_rnd, p, num_vars=10, alpha=100)

    # define neural architecture for the solution mapping
    func = nm.modules.blocks.MLP(insize=num_vars, outsize=num_vars, bias=True,
                                 linear_map=nm.slim.maps["linear"], nonlin=nn.ReLU, hsizes=[80]*4)
    # solution map from model parameters: sol_map(p) -> x
    sol_map = nm.system.Node(func, ["p"], ["x_bar"], name="smap")

    # round x
    round_func = roundModel(param_keys=["p"], var_keys=["x_bar"], output_keys=["x_rnd"],
                            int_ind={"x_bar":model.intInd}, input_dim=num_vars*2, hidden_dims=[80]*2, output_dim=num_vars,
                            name="round")
    #round_func = roundGumbelModel(param_keys=["p"], var_keys=["x_bar"], output_keys=["x_rnd"],
    #                              int_ind={"x_bar":model.intInd}, input_dim=num_vars*2, hidden_dims=[80]*4, output_dim=num_vars,
    #                              name="round")

    # proj x to feasible region
    num_steps = 5
    step_size = 0.1
    decay = 0.1
    proj = solPredGradProj(constraints=constrs_rnd,  # inequality constraints to be corrected
                           input_dim=num_vars*3, # dimension of input feature
                           hidden_dims=[80]*4, # dimensions of hidden layers
                           output_dim=num_vars, # dimension of output
                           param_keys=["p"], # model parameters
                           var_keys=["x_rnd"], # primal variables to be updated
                           output_keys=["x_bar"], # updated primal variables
                           num_steps=num_steps,  # number of rollout steps of the solver method
                           step_size=step_size,  # step size of the solver method
                           decay=decay,  # decay factor of the step size
                           name="proj")
    # solution distance
    f = sum((x_bar[:, i] - x_rnd[:, i]) ** 2 for i in range(num_vars))
    sol_dist = [f.minimize(weight=1.0, name="obj")]

    # trainable components
    components = [sol_map, round_func, proj]

    # penalty loss
    loss = nm.loss.PenaltyLoss(obj_rnd, constrs_rnd) + nm.loss.PenaltyLoss(sol_dist, constrs_bar)
    problem = nm.problem.Problem(components, loss, grad_inference=True)

    # training
    lr = 0.001    # step size for gradient descent
    epochs = 400  # number of training epochs
    warmup = 50   # number of epochs to wait before enacting early stopping policy
    patience = 50 # number of epochs with no improvement in eval metric to allow before early stopping
    # set adamW as optimizer
    optimizer = torch.optim.AdamW(problem.parameters(), lr=lr)
    # define trainer
    trainer = nm.trainer.Trainer(problem, loader_train, loader_dev, loader_test,
                                 optimizer, epochs=epochs, patience=patience, warmup=warmup)
    best_model = trainer.train()
    print()

    # params
    p = np.random.uniform(1, 11, (3, num_vars))
    print("Parameters p:", list(p[0]))
    print()

    # get solution from Ipopt
    print("Ipopt:")
    model_rel = model.relax()
    model_rel.setParamValue(*p[0])
    xval, _ = test.solverTest(model_rel, solver="ipopt")

    # rounding
    print("Round:")
    test.heurTest(naive_round, model, xval)

    # get solution from neuroMANCER
    print("neuroMANCER:")
    datapoints = {"p": torch.tensor(p, dtype=torch.float32),
                  "name": "test"}
    test.nmTest(problem, datapoints, model, x_name="test_x_rnd")

    # get solution from Ipopt
    print("SCIP:")
    model.setParamValue(*p[0])
    test.solverTest(model, solver="scip")