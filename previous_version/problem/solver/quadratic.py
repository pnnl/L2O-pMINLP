"""
Parametric Mixed Integer Quadratic Programming
"""

from pyomo import environ as pe

from problem.solver.abcMINLP import abcParamModel

class exactQuadratic(abcParamModel):
    def __init__(self, n_vars=None, n_integers=0):
        super().__init__()
        # create model
        m = pe.ConcreteModel()
        # parameters
        m.p = pe.Param(pe.RangeSet(0, n_vars-1), default=0, mutable=True)
        # variables
        m.x = {}
        # reorder the variable index
        varind = [i for i in range(n_vars) if i % 2 == 0] + [i for i in range(n_vars) if i % 2 != 0]
        # discrete
        m.x_int = pe.Var(pe.Set(initialize=varind[:n_integers]), domain=pe.Integers)
        # continuous
        m.x_real = pe.Var(pe.Set(initialize=varind[n_integers:]), domain=pe.Reals)
        # all variables
        for i in range(n_vars):
            if n_integers and i in m.x_int:
                m.x[i] = m.x_int[i]
            if i in m.x_real:
                m.x[i] = m.x_real[i]
        # objective
        obj = sum(m.x[i] ** 2 for i in m.x)
        m.obj = pe.Objective(sense=pe.minimize, expr=obj)
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(n_vars-1):
            m.cons.add(m.x[i] + m.x[(i+1)] - m.p[i] >= 0)
            m.cons.add(m.x[i] + m.x[(i+1)] - m.p[i] <= 5)
        m.cons.add(m.x[n_vars-1] - m.x[0] - m.p[n_vars-1] >= 0)
        m.cons.add(m.x[n_vars-1] - m.x[0] - m.p[n_vars-1] <= 5)
        # attribute
        self.model = m
        self.params ={"p":m.p}
        self.x = m.x
        self.cons = m.cons


if __name__ == "__main__":

    from utlis import test

    # params
    p = 3, 5, 7

    # init model
    model = exactQuadratic(n_vars=3, n_integers=2)

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