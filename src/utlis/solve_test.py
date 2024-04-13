"""
Test functions to solve problem
"""

import time

def ms_test_solve(model, tee=False):
    """
    Test function for mathematic solver
    """
    tick = time.time()
    solvals, objval = model.solve(tee=tee)
    tock = time.time()
    print("Binary Variables:", model.bin_ind)
    print("Integer Variables:", model.int_ind)
    for k, v in solvals.items():
        for i, val in v.items():
            print("{}[{}]: {:.2f}".format(k, i, val), end=" ")
    print("\nObjective Value: {:.2f}".format(objval))
    print("Constraint Violations: {:.4f}".format(sum(model.cal_violation())))
    print("Elapsed Time: {:.4f} sec".format(tock - tick))
    print()
    return solvals, objval


def nm_test_solve(var_keys, problem, datapoint, model):
    """
    Test function for neuroMANCER
    """
    # inference
    problem.eval()
    tick = time.time()
    output = problem(datapoint)
    tock = time.time()
    # get values
    for k1, k2 in zip(model.vars, var_keys):
        k2 = datapoint["name"] + "_" + k2
        for i in model.vars[k1]:
            model.vars[k1][i].value = output[k2][0, i].item()
    solvals, objval = model.get_val()
    # results
    print("Binary Variables:", model.bin_ind)
    print("Integer Variables:", model.int_ind)
    for k, v in solvals.items():
        for i, val in v.items():
            print("{}[{}]: {:.2f}".format(k, i, val), end=" ")
    print("\nObjective Value: {:.2f}".format(objval))
    print("Constraint Violations: {:.4f}".format(sum(model.cal_violation())))
    print("Elapsed Time: {:.4f} sec".format(tock - tick))
    print()
    return solvals, objval
