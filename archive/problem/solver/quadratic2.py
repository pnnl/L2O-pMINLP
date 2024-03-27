"""
Parametric Mixed Integer Quadratic Programming
"""

from pyomo import environ as pe

from problem.solver.abcMINLP import abcParamModel

class exactQuadratic2(abcParamModel):
    def __init__(self, c, Q, d, b, A, E, F):
        super().__init__()
        # create model
        m = pe.ConcreteModel()
        # parameters
        m.p = pe.Param(pe.RangeSet(0, 1), default=0, mutable=True)
        # decision variables
        vars = {}
        # continuous variables
        m.x = pe.Var(range(2), domain=pe.NonNegativeReals)
        for i in range(2):
            vars[i] = m.x[i]
        # binary variables
        m.y = pe.Var(range(2), domain=pe.Integers)
        for i in range(2):
            vars[i+2] = m.y[i]
        # objective
        obj = sum(c[i] * m.x[i] for i in range(2)) \
            + 0.5 * sum(Q[i, j] * m.x[i] * m.x[j] for i in range(2) for j in range(2)) \
            + sum(d[i] * m.y[i] for i in range(2))
        m.obj = pe.Objective(sense=pe.minimize, expr=obj)
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(7):
            lhs = sum(A[i, j] * m.x[j] for j in range(2)) + sum(E[i, j] * m.y[j] for j in range(2))
            rhs = b[i] + F[i,0] * m.p[0] + F[i,1] * m.p[1]
            m.cons.add(lhs <= rhs)
        # attribute
        self.model = m
        self.params ={"p":m.p}
        self.x = vars
        self.cons = m.cons


if __name__ == "__main__":

    import numpy as np
    from utlis import test

    # coefficients
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

    # params
    p = 0.5, 0.1

    # init model
    model = exactQuadratic2(c, Q, d, b, A, E, F)

    # relaxed
    print()
    print("======================================================")
    print("Real domain:")
    model_rel = model.relax()
    model_rel.setParamValue(*p)
    # ipopt
    test.solverTest(model_rel, "ipopt")
    # scip
    test.solverTest(model_rel, "scip")

    # penalty
    print()
    print("======================================================")
    print("Penalty:")
    model_pen = model.penalty(100)
    model_pen.setParamValue(*p)
    # scip
    test.solverTest(model_pen, "scip")

    # integer
    print()
    print("======================================================")
    print("Integer domain:")
    model.setParamValue(*p)
    # scip
    xval, _ = test.solverTest(model, "scip")