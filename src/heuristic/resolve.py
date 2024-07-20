import math

from pyomo import environ as pe

def rens(xval, model, tolerance=1e-5):
    # clone model
    model_rens = model.clone()
    # get solution value
    for k, vals in xval.items():
        # go through vars
        for i in vals:
            # skip continuous
            if model_rens.vars[k][i].domain == pe.Reals:
                continue
            # fix integer variable in relaxation
            if abs(xval[k][i] - round(xval[k][i])) <= tolerance:
                model_rens.vars[k][i].fix(round(xval[k][i]))
            # change to binary bounds
            else:
                model_rens.vars[k][i].setlb(math.floor(xval[k][i]))
                model_rens.vars[k][i].setub(math.ceil(xval[k][i]))
    # solve
    xval, objval = model_rens.solve("scip")
    return xval, objval

if __name__ == "__main__":

    from src.problem import msRosenbrock

    steepness = 50    # steepness factor
    num_blocks = 2    # number of expression blocks
    timelimit = 60    # time limit

    # init model
    model = msRosenbrock(steepness=steepness, num_blocks=num_blocks, timelimit=timelimit)

    # params
    p, a = 3.2, (2.4, 1.8)
    params = {"p":p, "a":a}
    # set params
    model.set_param_val({"p":p, "a":a})
    # relax model
    model_rel = model.relax()
    # solve relaxation
    xval_rel, _ = model_rel.solve("scip")
    print(xval_rel)
    # rens
    xval, objval = rens(xval_rel, model)
