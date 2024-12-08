import math
import heapq

import numpy as np
from pyomo import environ as pe

# ignore warning
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

def feasibility_pump(xval, model, perturbation=10, max_iters=20, seed=42):
    """
    A method for feasibility pump
    """
    # random state
    rnd = np.random.RandomState(seed)
    # init x_rel & x_int
    xval_rel = xval
    xval_int = roundStep(xval_rel, model)
    # iterate with limited steps
    for k in range(max_iters):
        # projection step
        xval_rel = projStep(xval_int, model)
        # feasible integer
        if is_integer(xval_rel, model):
            addVal(xval_rel, model)
            return
        # rounding step
        xval_round = roundStep(xval_rel, model)
        # update x_int with perturbation (avoid cycling)
        if not int_eqal(xval_int, xval_round, model):
            xval_int = xval_round
        else:
            num_perturbed = rnd.randint(math.floor(perturbation/2), math.ceil(3*perturbation/2)+1)
            sigma_score = {ind:abs(xval_rel[ind] - xval_int[ind]) for ind in xval_rel if model.x[ind].is_integer()}
            # n largest value
            largest_ind = heapq.nlargest(num_perturbed, sigma_score, key=sigma_score.get)
            # flip
            for ind in largest_ind:
                xval_int[ind] += np.sign(xval_rel[ind] - xval_int[ind])
    # assign integer solution
    addVal(xval_int, model)


def is_integer(xval, model):
    for ind in xval:
        if model.x[ind].is_integer() and not xval[ind].is_integer():
            return False
    return True


def addVal(xval, model):
    for ind in xval:
        model.x[ind].value = xval[ind]


def roundStep(xval_rel, model):
    xval_int = {}
    for ind in xval_rel:
        if model.x[ind].is_integer():
            xval_int[ind] = float(round(xval_rel[ind]))
        else:
            xval_int[ind] = xval_rel[ind]
    return xval_int


def projStep(xval_int, model):
    # relax to continuous model
    model_rel = model.relax()
    # delete original obj
    model_rel.model.del_component(model_rel.model.obj)
    # l2 distance on integer variables
    dist = sum((model_rel.x[ind] - xval_int[ind]) ** 2 for ind in xval_int if model.x[ind].is_integer())
    model_rel.model.obj = pe.Objective(expr=dist)
    # solve
    xval_rel, _ = model_rel.solve("scip")
    return xval_rel


def int_eqal(xval_int, xval_round, model):
    for ind in xval_int:
        if model.x[ind].is_integer():
            if int(xval_int[ind]) != int(xval_round[ind]):
                return False
    return True


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
    print("Feasibility Pump:")
    model = exactQuadratic(n_vars=10, n_integers=5)
    model.setParamValue(*p)
    xval, _ = test.heurTest(feasibility_pump, model, xval, perturbation=2, max_iters=10)


    # integers
    print("Integers:")
    model = exactQuadratic(n_vars=10, n_integers=5)
    model.setParamValue(*p)
    xval, _ = test.solverTest(model, solver="scip")