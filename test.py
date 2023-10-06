import time

import numpy as np

from problem.solver import exactQuadratic, exactRosenbrock, exactRastrigin
from heuristic import naive_round, feasibility_round, random_round, local_branch, rens, feasibility_pump

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # configuration
    parser.add_argument("--prob",
                        type=str,
                        default="qd",
                        choices=["qd", "rb", "rt"],
                        help="optimization problem")
    parser.add_argument("--varnum",
                        type=int,
                        default=20,
                        help="number of variables")
    parser.add_argument("--intnum",
                        type=int,
                        default=10,
                        help="number of integer variables")
    parser.add_argument("--algo",
                        type=str,
                        default="fp",
                        choices=["nr", "fr", "rr", "lb", "rens", "fp"],
                        help="correction algorithm")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed")

    # get configuration
    config = parser.parse_args()

    # random seed
    np.random.seed(config.seed)

    # quadratic solver
    if config.prob == "qd":
        print("Mixed Integer Quadratic Problem")
        # params
        p = np.random.uniform(1, 11, config.varnum)
        print("p:", p)
        # model
        model = exactQuadratic(*p, n_integers=config.intnum)
    # Rosenbrock
    if config.prob == "rb":
        print("Mixed Integer Rosenbrock Problem")
        # params
        p = np.random.uniform(0.5, 6.0)
        print("p:", p)
        a = np.random.uniform(0.2, 1.2, config.varnum-1)
        print("a:", a)
        # model
        model = exactRosenbrock(p, *a, n_integers=config.intnum)
    # Rastrigin
    if config.prob == "rt":
        print("Mixed Integer Rastrigin Problem")
        # params
        p = np.random.uniform(2, 6)
        print("p:", p)
        a = np.random.uniform(6, 15, config.varnum)
        print("a:", a)
        # model
        model = exactRastrigin(p, *a, n_integers=config.intnum)
    print()


    # integers
    print("Exact Integers:")
    tick = time.time()
    xval, objval = model.clone().solve("scip")
    tock = time.time()
    for i, val in xval.items():
        print("x[{}]: {:.2f}".format(i, val), end=" ")
    print("\nObjective Value: {:.2f}".format(objval))
    print("Elapsed Time: {:.4f} sec".format(tock - tick))
    print()

    # relaxation
    print("Continuous Relaxation:")
    model_rel = model.relax()
    tick = time.time()
    xval, objval = model_rel.solve("ipopt")
    tock = time.time()
    for i, val in xval.items():
        print("x[{}]: {:.2f}".format(i, val), end=" ")
    print("\nObjective Value: {:.2f}".format(objval))
    print("Elapsed time: {:.4f} sec".format(tock - tick))
    print()

    # heuristic
    print("Heuristic:")
    tick = time.time()
    if config.algo == "nr":
        print("Naive Round:")
        naive_round(xval, model)
    if config.algo == "fr":
        print("Feasible Round:")
        feasibility_round(xval, model)
    if config.algo == "rr":
        print("Random Round:")
        random_round(xval, model, seed=config.seed)
    if config.algo == "lb":
        print("Local Branch:")
        local_branch(xval, model, neighborhood=5, max_iters=1)
    if config.algo == "rens":
        print("RENS:")
        rens(xval, model)
    if config.algo == "fp":
        print("Feasibility Pump:")
        feasibility_pump(xval, model, perturbation=config.intnum//2, max_iters=5)
    tock = time.time()
    # get values
    xval, objval = model.getVal()
    for i, val in xval.items():
        print("x[{}]: {:.2f}".format(i, val), end=" ")
    print("\nObjective Value: {:.2f}".format(objval))
    print("Constraint Violations: {:.4f}".format(sum(model.calViolation())))
    print("Elapsed Time: {:.4f} sec".format(tock - tick))
    print()