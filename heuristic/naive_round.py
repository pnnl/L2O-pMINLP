# ignore warning
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

def naive_round(xval, model):
    """
    A method to just round
    """
    # assign initial solution
    for ind in xval:
        model.x[ind].value = xval[ind]
    # round
    for ind in xval:
        # get the variable's value and round it for less precision
        val = round(xval[ind], ndigits=7)
        # skip if the domain is not integer
        if not model.x[ind].is_integer():
            continue
        model.x[ind].value = round(val)


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
    print("Naive Rounding:")
    model = exactQuadratic(n_vars=10, n_integers=5)
    model.setParamValue(*p)
    xval, _ = test.heurTest(naive_round, model, xval)


    # integers
    print("Integers:")
    model = exactQuadratic(n_vars=10, n_integers=5)
    model.setParamValue(*p)
    xval, _ = test.solverTest(model, solver="scip")