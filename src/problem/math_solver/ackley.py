"""
Parametric Mixed Integer Constrained Ackley Problem
"""

import math
import numpy as np
from pyomo import environ as pe

from src.problem.math_solver import abcParamSolver

class ackley(abcParamSolver):
    def __init__(self, num_var, num_ineq, timelimit=None):
        super().__init__(timelimit=timelimit)
        # create model
        m = pe.ConcreteModel()
        # parameters
        m.b = pe.Param(pe.RangeSet(0, num_ineq-1), default=0, mutable=True)
        # variables
        m.x = pe.Var(pe.RangeSet(0, num_var-1), domain=pe.Integers)
        # objective
        norm_term = pe.sqrt(sum(m.x[i]**2 for i in range(num_var)) / num_var)
        cos_term = sum(pe.cos(2 * math.pi * m.x[i]) for i in range(num_var)) / num_var
        obj = -20 * pe.exp(-0.2 * norm_term) - pe.exp(cos_term) + 20 + math.e
        m.obj = pe.Objective(sense=pe.minimize, expr=obj)
        # constraints
        m.cons = pe.ConstraintList()
        rng = np.random.RandomState(17)
        Q = 0.01 * np.diag(rng.random(size=num_var)) # not used
        p = 0.1 * rng.random(num_var) # not used
        A = rng.normal(scale=0.1, size=(num_ineq, num_var))
        for i in range(num_ineq):
            m.cons.add(sum(A[i,j] * m.x[j] for j in range(num_var)) <= m.b[i])
        # attribute
        self.model = m
        self.params ={"b":m.b}
        self.vars = {"x":m.x}
        self.cons = m.cons


if __name__ == "__main__":

    from src.utlis import ms_test_solve

    num_var = 5
    num_ineq = 5
    timelimit = 60    # time limit

    # params
    b = np.random.uniform(-1, 1, size=num_ineq)
    params = {"b":b}
    # init model
    model = ackley(num_var=num_var, num_ineq=num_ineq, timelimit=timelimit)

    # solve the MIQP
    print("======================================================")
    print("Solve MINLP problem:")
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
