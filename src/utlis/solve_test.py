"""
Test functions to solve problem
"""

import time

def msSolveTest(model, tee=False):
    """
    Test function for mathematic solver
    """
    tick = time.time()
    solvals, objval = model.solve(tee=tee)
    tock = time.time()
    print("Binary Variables:", model.binInd)
    print("Integer Variables:", model.intInd)
    for k, v in solvals.items():
        for i, val in v.items():
            print("{}[{}]: {:.2f}".format(k, i, val), end=" ")
    print("\nObjective Value: {:.2f}".format(objval))
    print("Constraint Violations: {:.4f}".format(sum(model.calViolation())))
    print("Elapsed Time: {:.4f} sec".format(tock - tick))
    print()
    return solvals, objval
