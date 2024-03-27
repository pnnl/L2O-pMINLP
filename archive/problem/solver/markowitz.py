"""
Parametric Markowitz Portfolio Optimization
"""

from pyomo import environ as pe

from problem.solver.abcMINLP import abcParamModel

class exactMarkowitz(abcParamModel):
    def __init__(self, exp_returns, cov_matrix):
        super().__init__()
        # number of variables
        self.n_vars = len(exp_returns)
        # create model
        m = pe.ConcreteModel()
        # set index
        m.assets = pe.Set(initialize=range(self.n_vars))
        # parameters
        m.p = pe.Param(default=0, mutable=True)
        # variables
        m.x = pe.Var(m.assets, domain=pe.NonNegativeIntegers)
        # objective
        obj = sum(cov_matrix[i,j] * m.x[i] * m.x[j] for i in m.assets for j in m.assets)
        m.obj = pe.Objective(sense=pe.minimize, expr=obj)
        # constraints
        m.cons = pe.ConstraintList()
        # constr: 100 units
        m.cons.add(sum(m.x[i] for i in m.assets) == 100)
        # constr: expected return
        m.cons.add(sum(exp_returns[i] * m.x[i] for i in m.assets) >= 100 * m.p)
        # attribute
        self.model = m
        self.params ={"p":m.p}
        self.x = m.x
        self.cons = m.cons


if __name__ == "__main__":

    import numpy as np
    from utlis import test

    n = 5
    # expected returns
    exp_returns = np.random.uniform(0.002, 0.01, n)
    print("Expected Returns:")
    print(exp_returns)
    # covariance matrix
    A = np.random.rand(n,n)
    # positive semi-definite matrix
    cov_matrix = A @ A.T / 1000
    print("Covariance Matrix:")
    print(cov_matrix)

    # params
    p = 0.004

    # init model
    model = exactMarkowitz(exp_returns, cov_matrix)

    # relaxed
    print()
    print("======================================================")
    print("Real domain:")
    model_rel = model.relax()
    model_rel.setParamValue(p)
    # ipopt
    test.solverTest(model_rel, "ipopt")
    # scip
    test.solverTest(model_rel, "scip")

    # penalty
    print()
    print("======================================================")
    print("Penalty:")
    model_pen = model.penalty(100)
    model_pen.setParamValue(p)
    # scip
    test.solverTest(model_pen, "scip")

    # integer
    print()
    print("======================================================")
    print("Integer domain:")
    model.setParamValue(p)
    # scip
    xval, _ = test.solverTest(model, "scip")