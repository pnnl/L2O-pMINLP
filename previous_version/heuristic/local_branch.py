import numpy as np
from pyomo import environ as pe

# ignore warning
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

def local_branch(xval, model, neighborhood=10, max_iters=100):
    """
    A method for local branching
    """
    # init sol
    xval_best, objval_best = xval, float("inf")
    for k in range(max_iters):
        # add local branch constr
        model_cur = addLocalBranchConstr(model, xval_best, neighborhood)
        # get sol
        xval, objval = model_cur.solve("scip")
        # check improved
        if objval_best > objval:
            xval_best, objval_best = xval, objval
    addVal(xval_best, model)


def addLocalBranchConstr(model, xval_best, neighborhood):
    model = model.clone()
    # integer index
    ind_int = getIntVar(model)
    # auxiliary variables
    model.model.z = pe.Var(ind_int, within=pe.NonNegativeReals)
    model.z = model.model.z
    # add constraints
    for ind in ind_int:
        model.cons.add(model.z[ind] >= model.x[ind] - xval_best[ind])
        model.cons.add(model.z[ind] >= xval_best[ind] - model.x[ind])
    model.cons.add(sum(model.z[ind] for ind in ind_int) <= neighborhood)
    return model


def getIntVar(model):
    ind_int = []
    for ind in model.x:
        if model.x[ind].domain == pe.Integers:
            ind_int.append(ind)
    return ind_int


def addVal(xval, model):
    for ind in xval:
        model.x[ind].value = xval[ind]


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
    print("Local Branch:")
    model = exactQuadratic(n_vars=10, n_integers=5)
    model.setParamValue(*p)
    xval, _ = test.heurTest(local_branch, model, xval, neighborhood=25, max_iters=3)


    # integers
    print("Integers:")
    model = exactQuadratic(n_vars=10, n_integers=5)
    model.setParamValue(*p)
    xval, _ = test.solverTest(model, solver="scip")