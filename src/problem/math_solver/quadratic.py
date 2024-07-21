"""
Parametric Mixed Integer Quadratic Programming

https://www.sciencedirect.com/science/article/pii/S0098135401007979
https://www.sciencedirect.com/science/article/pii/S1570794601801577
"""

import numpy as np
from pyomo import environ as pe

from src.problem.math_solver import abcParamSolver

class quadratic(abcParamSolver):
    def __init__(self, timelimit=None):
        super().__init__(timelimit=timelimit)
        # define the coefficients for the quadratic problem
        c = np.array([[0.0200],
                      [0.0300]])
        Q = np.array([[0.0196, 0.0063],
                      [0.0063, 0.0199]])
        d = np.array([[-0.3000],
                      [-0.3100]])
        b = np.array([[0.417425],
                      [3.582575],
                      [0.413225],
                      [0.467075],
                      [1.090200],
                      [2.909800],
                      [1.000000]])
        A = np.array([[ 1.0000,  0.0000],
                      [-1.0000,  0.0000],
                      [-0.0609,  0.0000],
                      [-0.0064,  0.0000],
                      [ 0.0000,  1.0000],
                      [ 0.0000, -1.0000],
                      [ 0.0000,  0.0000]])
        E = np.array([[-1.0000,  0.0000],
                      [-1.0000,  0.0000],
                      [ 0.0000, -0.5000],
                      [ 0.0000, -0.7000],
                      [-0.6000,  0.0000],
                      [-0.5000,  0.0000],
                      [ 1.0000,  1.0000]])
        F = np.array([[ 3.16515,  3.7546],
                      [-3.16515, -3.7546],
                      [ 0.17355, -0.2717],
                      [ 0.06585,  0.4714],
                      [ 1.81960, -3.2841],
                      [-1.81960,  3.2841],
                      [ 0.00000,  0.0000]])
        # create model
        m = pe.ConcreteModel()
        # mutable parameters (parametric part of the problem)
        m.p = pe.Param(pe.RangeSet(0, 1), default=0, mutable=True)
        # decision variables
        domains = [pe.NonNegativeReals] * 2 + [pe.Binary] * 2
        m.x = pe.Var(range(4), domain=lambda m, i: pe.NonNegativeReals if i < 2 else pe.Binary)
        # objective function C^T x + 1/2 x^T Q x + d^T y
        obj = sum(c[i] * m.x[i] for i in range(2)) \
            + 0.5 * sum(Q[i, j] * m.x[i] * m.x[j] for i in range(2) for j in range(2)) \
            + sum(d[i] * m.x[i+2] for i in range(2))
        m.obj = pe.Objective(sense=pe.minimize, expr=obj)
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(7):
            # LHS: Ax + Ey
            lhs = sum(A[i, j] * m.x[j] for j in range(2)) \
                + sum(E[i, j] * m.x[j+2] for j in range(2))
            # RHS: b + FP
            rhs = b[i] + F[i,0] * m.p[0] + F[i,1] * m.p[1]
            # Ax + Ey <= b + FP
            m.cons.add(lhs <= rhs)
        # set attributes
        self.model = m
        self.params ={"p":m.p}
        self.vars = {"x":m.x}
        self.cons = m.cons

if __name__ == "__main__":

    from src.utlis import ms_test_solve

    # set params
    p = 0.6, 0.8
    params = {"p":p}
    # init model
    model = quadratic()

    # solve the MIQP
    print("======================================================")
    print("Solve MINLP problem:")
    model.set_param_val(params)
    ms_test_solve(model)

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
