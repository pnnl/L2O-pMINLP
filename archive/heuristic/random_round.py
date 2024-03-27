import math

# ignore warning
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

def random_round(xval, model, seed=42):
    """
    A random method to round fractional variables in feasible region
    """
    # assign initial solution
    for ind in xval:
        model.x[ind].value = xval[ind]
    # random state
    rnd = np.random.RandomState(seed)
    # variable index
    var_inds = list(model.x.keys())
    # init counter
    k = 0
    while var_inds:
        # pick a variable randomly
        ind = rnd.choice(var_inds)
        var_inds.remove(ind)
        # get the variable's value and round it for less precision
        val = round(model.x[ind].value, ndigits=7)
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
        # random round
        if rnd.rand() < floor_violation / (floor_violation + ceil_violation):
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
    print("random Rounding:")
    model = exactQuadratic(n_vars=10, n_integers=5)
    model.setParamValue(*p)
    xval, _ = test.heurTest(random_round, model, xval, seed=42)


    # integers
    print("Integers:")
    model = exactQuadratic(n_vars=10, n_integers=5)
    model.setParamValue(*p)
    xval, _ = test.solverTest(model, solver="scip")