import time

def solverTest(model, solver="ipopt"):
    tick = time.time()
    xval, objval = model.solve(solver, max_iter=100)
    tock = time.time()
    for i, val in xval.items():
        print("x[{}]: {:.2f}".format(i, val), end=" ")
    print("\nObjective Value: {:.2f}".format(objval))
    print("Constraint Violations: {:.4f}".format(sum(model.calViolation())))
    print("Elapsed Time: {:.4f} sec".format(tock - tick))
    print()
    return xval, objval


def heurTest(heur_func, model, xval_init, **kwargs):
    tick = time.time()
    heur_func(xval_init, model, **kwargs)
    tock = time.time()
    # get values
    xval, objval = model.getVal()
    for i, val in xval.items():
        print("x[{}]: {:.2f}".format(i, val), end=" ")
    print("\nObjective Value: {:.2f}".format(objval))
    print("Constraint Violations: {:.4f}".format(sum(model.calViolation())))
    print("Elapsed time: {:.4f} sec".format(tock - tick))
    print()
    return xval, objval


def nmTest(problem, datapoints, model, x_name="test_x"):
    # inference
    problem.eval()
    tick = time.time()
    output = problem(datapoints)
    x = output[x_name]
    tock = time.time()
    # get values
    for ind in model.x:
        model.x[ind].value = x[0, ind].item()
    xval, objval = model.getVal()
    for i, val in xval.items():
        print("x[{}]: {:.2f}".format(i, val), end=" ")
    print("\nObjective Value: {:.2f}".format(objval))
    print("Constraint Violations: {:.4f}".format(sum(model.calViolation())))
    print("Elapsed Time: {:.4f} sec".format(tock - tick))
    print()
    return xval, objval


if __name__ == "__main__":

    import numpy as np
    import neuromancer as nm
    import torch
    from torch import nn

    from problem.solver import exactQuadratic
    from problem.neural import nnQuadratic

    # random seed
    np.random.seed(42)

    # number of variables
    num_vars = 10

    # random parameter
    p = np.random.uniform(1, 11, num_vars)
    print("Parameters p:", list(p))

    # get solution from Ipopt
    print("Ipopt:")
    model = exactQuadratic(num_vars, n_integers=0)
    model.setParamValue(*p)
    solverTest(model, solver="ipopt")

    print("SCIP:")
    model = exactQuadratic(num_vars, n_integers=5)
    model.setParamValue(*p)
    solverTest(model, solver="scip")

    print("Feasibility Pump:")
    model = exactQuadratic(num_vars, n_integers=5)
    model.setParamValue(*p)
    xval = {0:-1.83, 1:6.57, 2:3.93, 3:4.39, 4:2.60, 5:1.28, 6:1.28, 7:4.83, 8:4.83, 9:6.25}
    from heuristic import feasibility_pump
    heurTest(feasibility_pump, model, xval, perturbation=2, max_iters=10)

    print("neuroMANCER:")
    # define neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_vars, outsize=num_vars, bias=True,
                                 linear_map=nm.slim.maps["linear"], nonlin=nn.ReLU, hsizes=[50]*4)
    # get solver
    problem = nnQuadratic(num_vars, func, alpha=100)
    datapoints = {"p": torch.tensor([list(p)], dtype=torch.float32),
                  "name": "test"}
    model = exactQuadratic(num_vars, n_integers=5)
    model.setParamValue(*p)
    nmTest(problem, datapoints, model)