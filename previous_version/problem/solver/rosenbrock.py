"""
Parametric Mixed Integer Constrained Rosenbrock Problem
"""

from pyomo import environ as pe

from problem.solver.abcMINLP import abcParamModel

class exactRosenbrock(abcParamModel):
    def __init__(self, n_vars, n_integers=0):
        super().__init__()
        # create model
        m = pe.ConcreteModel()
        # parameters
        m.p = pe.Param(pe.RangeSet(0, 0), default=0, mutable=True)
        m.a = pe.Param(pe.RangeSet(0, n_vars-2), default=0, mutable=True)
        # variables
        m.x = {}
        # discrete
        m.x_int = pe.Var(pe.RangeSet(0, n_integers-1), domain=pe.Integers)
        for i in range(n_integers):
            m.x[i] = m.x_int[i]
        # continuous
        m.x_real = pe.Var(pe.RangeSet(n_integers, n_vars-1), domain=pe.Reals)
        for i in range(n_integers, n_vars):
            m.x[i] = m.x_real[i]
        # objective
        obj = sum((1 - m.x[i]) ** 2 + m.a[i] * (m.x[i+1] - m.x[i] ** 2) ** 2 for i in range(n_vars-1))
        m.obj = pe.Objective(sense=pe.minimize, expr=obj)
        # constraints
        m.cons = pe.ConstraintList()
        m.cons.add(sum((-1)**i * m.x[i] for i in m.x) >= 0)
        m.cons.add(sum(m.x[i] ** 2 for i in m.x) >= m.p[0] / 2)
        m.cons.add(sum(m.x[i] ** 2 for i in m.x) <= m.p[0])
        # attribute
        self.model = m
        self.params ={"p":m.p, "a":m.a}
        self.x = m.x
        self.cons = m.cons

if __name__ == "__main__":

    from utlis import test

    # params
    p, a = 1.2, (0.4, 0.8)

    # init model
    model = exactRosenbrock(n_vars=3, n_integers=2)

    # relaxed
    print()
    print("======================================================")
    print("Real domain:")
    model_rel = model.relax()
    model_rel.setParamValue(p, *a)
    # ipopt
    test.solverTest(model_rel, "ipopt")
    # scip
    xval, _ = test.solverTest(model_rel, "scip")

    print()
    print("======================================================")
    print("Penalty:")
    model_pen = model.penalty(100)
    model_pen.setParamValue(p, *a)
    # scip
    test.solverTest(model_pen, "scip")


    # integer
    print()
    print("======================================================")
    print("Integer domain:")
    model.setParamValue(p, *a)
    # scip
    test.solverTest(model, "scip")