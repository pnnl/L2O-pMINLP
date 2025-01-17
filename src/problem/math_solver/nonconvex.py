"""
Parametric Mixed Integer Simple Conconvex Programming

https://arxiv.org/abs/2104.12225
"""

import numpy as np
from pyomo import environ as pe

from src.problem.math_solver import abcParamSolver

class nonconvex(abcParamSolver):
    def __init__(self, num_var, num_eq, num_ineq, timelimit=None):
        super().__init__(timelimit=timelimit, solver="scip")
        # fixed params
        rng = np.random.RandomState(17)
        self.Q = 0.01 * np.diag(rng.random(size=num_var))
        self.p = 0.1 * rng.random(num_var)
        self.A = rng.normal(scale=0.1, size=(num_eq, num_var))
        self.G = rng.normal(scale=0.1, size=(num_ineq, num_var))
        self.h = np.sum(np.abs(self.G @ np.linalg.pinv(self.A)), axis=1)
        # size
        num_ineq, num_var = self.A.shape
        # create model
        m = pe.ConcreteModel()
        # mutable parameters (parametric part of the problem)
        m.b = pe.Param(pe.RangeSet(0, num_ineq-1), default=0, mutable=True)
        m.d = pe.Param(pe.RangeSet(0, num_ineq-1), default=0, mutable=True)
        # decision variables
        m.x = pe.Var(range(num_var), domain=pe.Reals)
        for j in range(num_var // 2):
            m.x[j].domain = pe.Integers
        # objective function 1/2 x^T Q x + p^T sin(x)
        obj = sum(m.x[j] * self.Q[j,j] * m.x[j] / 2 + self.p[j] * pe.sin(m.x[j]) for j in range(num_var))
        m.obj = pe.Objective(sense=pe.minimize, expr=obj)
        # constraints
        m.cons = pe.ConstraintList()
        # constraints A x == b
        for i in range(num_eq):
            m.cons.add(sum(self.A[i,j] * m.x[j] for j in range(num_var)) == m.b[i])
        # constraints G x <= h
        for i in range(num_ineq):
            m.cons.add(sum(self.G[i,j] * m.x[j] for j in range(num_var)) <= self.h[i])
        # set attributes
        self.model = m
        self.params ={"b":m.b}
        self.vars = {"x":m.x}
        self.cons = m.cons

if __name__ == "__main__":

    from src.utlis import ms_test_solve

    num_var = 100
    num_ineq = 50
    num_eq = 50
    num_data = 5000

    # generate parameters
    b = np.random.uniform(-1, 1, size=(num_data, num_eq))
    # set params
    params = {"b":b[0]}
    # init model
    model = nonconvex(num_var, num_eq, num_ineq, timelimit=60)

    # solve the nonconvex
    print("======================================================")
    print("Solve MINLP problem:")
    model.set_param_val(params)
    ms_test_solve(model, tee=True)

    # solve the penalty
    print()
    print("======================================================")
    print("Solve penalty problem:")
    model_pen = model.penalty(100)
    model_pen.set_param_val(params)
    # scip
    ms_test_solve(model_pen)

    # solve the relaxation
    print()
    print("======================================================")
    print("Solve relaxed problem:")
    model_rel = model.relax()
    model_rel.set_param_val(params)
    # scip
    ms_test_solve(model_rel)
