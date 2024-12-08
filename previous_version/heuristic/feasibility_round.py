import math

# ignore warning
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

def feasibility_round(xval, model):
    """
    A method to round fractional variables in feasible region
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
        # skip if the value is already an integer
        if val.is_integer():
            continue
        # try floor
        floor_value = float(math.floor(val))
        model.x[ind].value = floor_value
        floor_violation = sum(model.calViolation())
        # try ceil
        ceil_value = float(math.ceil(val))
        model.x[ind].value = ceil_value
        ceil_violation = sum(model.calViolation())
        # round with less violation
        if floor_violation < ceil_violation:
            model.x[ind].value = floor_value
        else:
            model.x[ind].value = ceil_value


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
    print("Feasibility Round:")
    model = exactQuadratic(n_vars=10, n_integers=5)
    model.setParamValue(*p)
    xval, _ = test.heurTest(feasibility_round, model, xval)


    # integers
    print("Integers:")
    model = exactQuadratic(n_vars=10, n_integers=5)
    model.setParamValue(*p)
    xval, _ = test.solverTest(model, solver="scip")