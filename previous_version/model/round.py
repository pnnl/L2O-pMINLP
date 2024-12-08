from torch import nn

from model.func import diffFloor, diffBinarize
from model.layer import layerFC

class roundModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, int_ind, continuous_update=False, tolerance=1e-3):
        super(roundModel, self).__init__()
        # index of integer variables
        self.int_ind = int_ind
        # update continuous or not
        self.continuous_update = continuous_update
        # floor
        self.floor = diffFloor()
        # sequence
        sizes = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(layerFC(sizes[i], sizes[i + 1]))
        self.layers = nn.Sequential(*layers)
        # binarize
        self.bin = diffBinarize()
        self.tolerance = tolerance

    def forward(self, p, x):
        # concatenate all features: params + sol
        f = torch.cat([p, x], dim=-1)
        # forward
        h = self.layers(f)
        # floor
        x_flr = self.floor(x[:,self.int_ind])
        # binary
        bnr = self.bin(h[:,self.int_ind])
        # mask if already integer
        bnr = self._intMask(bnr, x[:,self.int_ind])
        # update continuous variables
        x_rnd = x + self.continuous_update * h
        # learnable round down: floor + 0 / round up: floor + 1
        x_rnd[:,self.int_ind] = x_flr + bnr
        return x_rnd

    def _intMask(self, bnr, x):
        # difference
        diff_flr = torch.abs(x - torch.floor(x))
        diff_cl = torch.abs(x - torch.ceil(x))
        # mask
        bnr[diff_flr < self.tolerance] = 0.0
        bnr[diff_cl < self.tolerance] = 1.0
        return bnr


if __name__ == "__main__":

    import numpy as np
    import torch
    from torch import nn
    import neuromancer as nm

    from problem.solver import exactQuadratic
    from heuristic import feasibility_round
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

    # define neural architecture for the solution mapping
    func = nm.modules.blocks.MLP(insize=num_vars, outsize=num_vars, bias=True,
                                 linear_map=nm.slim.maps["linear"], nonlin=nn.ReLU, hsizes=[80]*4)
    # define solver for solution
    #from model.func import solverWrapper
    #func = solverWrapper(model)
    # solution map from model parameters: sol_map(p) -> x
    sol_map = nm.system.Node(func, ["p"], ["x_bar"], name="smap")

    # round x
    round_func = roundModel(input_dim=num_vars*2, hidden_dims=[80]*2, output_dim=num_vars, int_ind=model.intInd)
    l_round = nm.system.Node(round_func, ["p", "x_bar"], ["x_rnd"], name="round")

    # trainable components
    components = [sol_map, l_round]

    # penalty loss
    from problem.neural import lossQuadratic
    loss = lossQuadratic(x_rnd, p, num_vars=10, alpha=100) + 0.5 * lossQuadratic(x_bar, p, num_vars=10, alpha=100)
    problem = nm.problem.Problem(components, loss)

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
    test.heurTest(feasibility_round, model, xval)

    # get solution from neuroMANCER
    print("neuroMANCER:")
    datapoints = {"p": torch.tensor(p, dtype=torch.float32),
                  "name": "test"}
    test.nmTest(problem, datapoints, model, x_name="test_x_rnd")

    # get solution from Ipopt
    print("SCIP:")
    model.setParamValue(*p[0])
    test.solverTest(model, solver="scip")