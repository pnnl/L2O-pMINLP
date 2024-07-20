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
                model_rens.vars[k][i].setlb(math.ceil(xval[k][i]))
                model_rens.vars[k][i].setub(math.floor(xval[k][i]))
            # solve
            xval, objval = model_rens.solve("scip")
    return xval, objval
