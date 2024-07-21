"""
Parametric Multi-Dimensional Knapsack

https://link.springer.com/article/10.1007/s12532-024-00255-x
"""

import numpy as np
from pyomo import environ as pe

from src.problem.math_solver import abcParamSolver

class knapsack(abcParamSolver):
    def __init__(self, weights, caps, timelimit=None):
        super().__init__(solver="gurobi", timelimit=timelimit)
        # problem size
        dim, num_var = weights.shape
        # create model
        m = pe.ConcreteModel()
        # parameters
        m.c = pe.Param(pe.RangeSet(0, num_var-1), default=0, mutable=True)
        # variables
        m.x = pe.Var(pe.RangeSet(0, num_var-1), domain=pe.Binary)
        # objective
        obj = sum(m.c[j] * m.x[j] for j in range(num_var))
        m.obj = pe.Objective(sense=pe.maximize, expr=obj)
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(dim):
            m.cons.add(sum(weights[i,j] * m.x[j] for j in range(num_var)) <= caps[i])
        # attribute
        self.model = m
        self.params ={"c":m.c}
        self.vars = {"x":m.x}
        self.cons = m.cons


if __name__ == "__main__":

    import pyepo
    from src.utlis import ms_test_solve

    num_var = 32      # number of variables
    dim = 2           # dimension of constraints
    caps = [20] * dim # capacity
    timelimit = 60    # time limit

    # generate data
    weights, x, c = pyepo.data.knapsack.genData(num_data=1, num_features=5,
                                                num_items=num_var, dim=dim,
                                                deg=4, noise_width=0.5)
    params = {"c":c[0]}
    # init model
    model = knapsack(weights=weights, caps=caps, timelimit=timelimit)

    # solve the MIP
    print("======================================================")
    print("Solve MIP problem:")
    model.set_param_val(params)
    ms_test_solve(model, tee=False)


    # solve the relaxation
    print()
    print("======================================================")
    print("Solve relaxed problem:")
    model_rel = model.relax()
    model_rel.set_param_val(params)
    # scip
    ms_test_solve(model_rel, tee=False)
