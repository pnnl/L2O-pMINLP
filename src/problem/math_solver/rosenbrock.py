"""
Parametric Mixed Integer Constrained Rosenbrock Problem
"""

from pyomo import environ as pe

from src.problem.math_solver import abcParamSolver

class rosenbrock(abcParamSolver):
    def __init__(self, num_vars, num_integers=0):
        super().__init__()
        # create model
        m = pe.ConcreteModel()
        # parameters
        m.p = pe.Param(pe.RangeSet(0, 0), default=0, mutable=True)
        m.a = pe.Param(pe.RangeSet(0, num_vars-2), default=0, mutable=True)
        # variables
        m.x = {}
        # discrete
        m.x_int = pe.Var(pe.RangeSet(0, num_integers-1), domain=pe.Integers)
        for i in range(num_integers):
            m.x[i] = m.x_int[i]
        # continuous
        m.x_real = pe.Var(pe.RangeSet(num_integers, num_vars-1), domain=pe.Reals)
        for i in range(num_integers, num_vars):
            m.x[i] = m.x_real[i]
        # objective
        obj = sum((1 - m.x[i]) ** 2 + m.a[i] * (m.x[i+1] - m.x[i] ** 2) ** 2 for i in range(num_vars-1))
        m.obj = pe.Objective(sense=pe.minimize, expr=obj)
        # constraints
        m.cons = pe.ConstraintList()
        m.cons.add(sum((-1)**i * m.x[i] for i in m.x) >= 0)
        m.cons.add(sum(m.x[i] ** 2 for i in m.x) >= m.p[0] / 2)
        m.cons.add(sum(m.x[i] ** 2 for i in m.x) <= m.p[0])
        # attribute
        self.model = m
        self.params ={"p":m.p, "a":m.a}
        self.vars = {"x":m.x}
        self.cons = m.cons


if __name__ == "__main__":

    from src.utlis import ms_test_solve

    # params
    p, a = 1.2, (0.4, 0.8)
    params = {"p":p, "a":a}
    # init model
    model = rosenbrock(num_vars=3, num_integers=2)

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
