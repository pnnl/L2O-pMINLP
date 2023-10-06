import math

from pyomo import environ as pe

# ignore warning
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

def rens(xval, model, tolerance=1e-5):
    # all integer
    for ind in model.x:
        # skip continuous
        if model.x[ind].domain != pe.Integers:
            continue
        # fix integer variable in relaxation
        if abs(xval[ind] - round(xval[ind])) <= tolerance:
            model.x[ind].fix(round(xval[ind]))
        # change to binary bounds
        else:
            model.x[ind].setlb(math.ceil(xval[ind]))
            model.x[ind].setub(math.floor(xval[ind]))
    # solve
    model.solve("scip")

if __name__ == "__main__":

    import numpy as np

    from problem.solver import exactQuadratic
    from utlis import test

    # random seed
    np.random.seed(42)

    # params
    p = np.random.uniform(1, 11, 10)
    print("Parameters", list(p))
    print()

    # reals
    print("Continuous:")
    model = exactQuadratic(n_vars=10, n_integers=0)
    model.setParamValue(*p)
    xval, _ = test.solverTest(model, solver="ipopt")


    # heuristic
    print("RENS:")
    model = exactQuadratic(n_vars=10, n_integers=5)
    model.setParamValue(*p)
    xval, _ = test.heurTest(rens, model, xval)


    # integers
    print("Integers:")
    model = exactQuadratic(n_vars=10, n_integers=5)
    model.setParamValue(*p)
    xval, _ = test.solverTest(model, solver="scip")
