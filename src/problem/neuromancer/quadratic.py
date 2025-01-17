"""
Parametric Mixed Integer Quadratic Programming
"""

import numpy as np
import torch
from torch import nn
import neuromancer as nm
from neuromancer.modules.solvers import GradientProjection

class penaltyLoss(nn.Module):
    """
    Penalty loss function for quadratic problem
    """
    def __init__(self, input_keys, num_var, num_eq, num_ineq, penalty_weight=100, output_key="loss"):
        super().__init__()
        self.num_var = num_var
        self.b_key, self.x_key = input_keys
        self.output_key = output_key
        self.penalty_weight = penalty_weight
        self.device = None
        # fixed coefficients
        rng = np.random.RandomState(17)
        Q = 0.01 * np.diag(rng.random(size=num_var))
        p = 0.1 * rng.random(num_var)
        A = rng.normal(scale=0.1, size=(num_eq, num_var))
        G = rng.normal(scale=0.1, size=(num_ineq, num_var))
        h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1)
        self.Q = torch.from_numpy(Q).float()
        self.p = torch.from_numpy(p).float()
        self.A = torch.from_numpy(A).float()
        self.G = torch.from_numpy(G).float()
        self.h = torch.from_numpy(h).float()

    def forward(self, input_dict):
        """
        forward pass
        """
        # objective function
        obj = self.cal_obj(input_dict)
        # constraints violation
        viol = self.cal_constr_viol(input_dict)
        # penalized loss
        loss = obj + self.penalty_weight * viol
        input_dict[self.output_key] = torch.mean(loss)
        return input_dict

    def cal_obj(self, input_dict):
        """
        calculate objective function
        """
        # get values
        x, b = input_dict[self.x_key], input_dict[self.b_key]
        # update device
        if self.device is None:
            self.device = x.device
            self.Q = self.Q.to(self.device)
            self.p = self.p.to(self.device)
            self.A = self.A.to(self.device)
            self.G = self.G.to(self.device)
            self.h = self.h.to(self.device)
        # 1/2 x^T Q x
        Q_term = torch.einsum("bm,nm,bm->b", x, self.Q, x) / 2
        # p^T y
        p_term = torch.einsum("m,bm->b", self.p, x)
        return Q_term + p_term

    def cal_constr_viol(self, input_dict):
        """
        calculate constraints violation
        """
        # get values
        x, b = input_dict[self.x_key], input_dict[self.b_key]
        # eq constraints A x == b
        #lhs = torch.einsum("mn,bn->bm", self.A, x) # Ax
        #rhs = b
        #eq_violation = torch.relu(torch.abs(lhs - rhs) - 1e-6)
        # ineq constraints G x <= h
        lhs = torch.einsum("mn,bn->bm", self.G, x) # Gx
        rhs = self.h
        ineq_violation = torch.relu(lhs - rhs)
        return ineq_violation.sum(dim=1)


class augmentedLagrangianLoss(penaltyLoss):
    """
    Augmented Lagrangian loss function for quadratic problem
    """
    def __init__(self, input_keys, num_var, num_eq, num_ineq, penalty_weight=100, growth_rate=1.0005, output_key="loss"):
        super().__init__(input_keys, num_var, num_eq, num_ineq, penalty_weight, output_key)
        # Lagrangian multiplier
        self.multiplier = nn.Parameter(torch.zeros(num_ineq), requires_grad=False)
        # penalty growth rate
        self.growth_rate = growth_rate

    def forward(self, input_dict):
        """
        forward pass
        """
        # objective function
        obj = self.cal_obj(input_dict)
        # constraints loss
        constr = self.cal_constr(input_dict)
        # penalized loss
        loss = obj + constr
        input_dict[self.output_key] = torch.mean(loss)
        return input_dict

    def cal_constr(self, input_dict):
        """
        calculate inequality constraints loss
        """
        if self.multiplier.device != self.device:
            self.multiplier = self.multiplier.to(self.device)
        # get values
        x, b = input_dict[self.x_key], input_dict[self.b_key]
        # ineq constraints G x <= h
        lhs = torch.einsum("mn,bn->bm", self.G, x) # Gx
        rhs = self.h
        val = lhs - rhs
        # lagrangian term: λ * g_+(x)
        lagrangian_term = self.multiplier * val
        # penalty term: ρ / 2 * g_+(x)^2
        penalty_term = 0.5 * self.penalty_weight * torch.relu(val) ** 2
        with torch.no_grad():
            self.update_penalty_weight(val)
        return (lagrangian_term + penalty_term).sum(dim=1)

    @torch.no_grad()
    def update_penalty_weight(self, val):
        # update lagrangian multiplier
        self.multiplier.data = torch.relu(self.multiplier.data + self.penalty_weight * val.sum(dim=0))
        # update penalty weight
        self.penalty_weight *= self.growth_rate


if __name__ == "__main__":

    # random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # init
    num_var = 200
    num_eq = 20
    num_ineq = 100
    hlayers_sol = 5
    hlayers_rnd = 4
    hsize = 512
    batch_size = 64
    lr = 1e-3
    penalty_weight = 100
    num_data = 10000
    test_size = 1000
    val_size = 1000
    train_size = num_data - test_size - val_size

    # init mathmatic model
    from src.problem.math_solver.quadratic import quadratic
    model = quadratic(num_var, num_eq, num_ineq, timelimit=60)

    # data sample from uniform distribution
    b_samples = torch.from_numpy(np.random.uniform(-1, 1, size=(num_data, num_eq))).float()
    data = {"b":b_samples}
    # data split
    from src.utlis import data_split
    data_train, data_test, data_val = data_split(data, test_size=test_size, val_size=val_size)
    # torch dataloaders
    from torch.utils.data import DataLoader
    loader_train = DataLoader(data_train, batch_size, num_workers=0,
                              collate_fn=data_train.collate_fn, shuffle=True)
    loader_test  = DataLoader(data_test, batch_size, num_workers=0,
                              collate_fn=data_test.collate_fn, shuffle=False)
    loader_val   = DataLoader(data_val, batch_size, num_workers=0,
                              collate_fn=data_val.collate_fn, shuffle=False)

    # define neural architecture for the solution map smap(p) -> x
    import neuromancer as nm
    from src.func.layer import netFC
    func = netFC(input_dim=num_eq, hidden_dims=[hsize]*hlayers_sol, output_dim=num_var-num_eq)
    smap = nm.system.Node(func, ["b"], ["x"], name="smap")

    # define rounding model
    from src.func.layer import netFC
    from src.func import roundGumbelModel
    layers_rnd = netFC(input_dim=num_var, hidden_dims=[hsize]*hlayers_rnd,
                       output_dim=num_var-num_eq)
    rnd = roundGumbelModel(layers=layers_rnd, param_keys=["b"], var_keys=["x"],
                           output_keys=["x_rnd"], int_ind=model.int_ind,
                           continuous_update=False, name="round")

    # complete solution
    from src.func import completePartial
    complete = completePartial(A=torch.from_numpy(model.A).float(), num_var=num_var,
                               partial_ind=range(num_var-num_eq), var_key="x_rnd",
                               rhs_key="b", output_key="x_comp", name="comp")

    # build neuromancer components
    components = nn.ModuleList([smap, rnd, complete])

    # build neuromancer problem
    loss_fn = penaltyLoss(["b", "x_comp"], num_var, num_eq, num_ineq, penalty_weight)

    # training
    from src.problem.neuromancer.trainer import trainer
    epochs = 200                    # number of training epochs
    patience = 20                   # number of epochs with no improvement in eval metric to allow before early stopping
    warmup = 40                     # number of epochs to wait before enacting early stopping policies
    optimizer = torch.optim.AdamW(components.parameters(), lr=lr)
    # create a trainer for the problem
    my_trainer = trainer(components, loss_fn, optimizer, epochs=epochs,
                         patience=patience, warmup=warmup)
    # training for the rounding problem
    my_trainer.train(loader_train, loader_val)

    # test neuroMANCER
    from src.utlis import nm_test_solve
    print("======================================================")
    print("neuroMANCER:")
    datapoint = {"b": b_samples[:1],
                 "name":"test"}
    model.set_param_val({"b":b_samples[0].cpu().numpy()})
    nm_test_solve("x_comp", components, datapoint, model, tee=True)

    # solve the MIQP
    from src.utlis import ms_test_solve
    print("======================================================")
    print("Gurobi:")
    ms_test_solve(model, tee=True)
