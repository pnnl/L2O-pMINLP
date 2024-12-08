"""
Multi-parametric Mixed-integer Linear Programming (mpMILP) problem using Neuromancer:
minimize     c^T*x
subject to   g(x, p) <= 0

problem parameters:            p, c
problem decition variables:    x in N
"""
import warnings
warnings.filterwarnings("ignore")

import torch
import slim
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import cvxpy as cp
import numpy as np
import time

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
import neuromancer.arg as arg
from neuromancer.constraint import variable
from neuromancer.activations import activations
from neuromancer.loggers import BasicLogger, MLFlowLogger
from neuromancer.dataset import get_static_dataloaders
from neuromancer.loss import get_loss
from neuromancer.maps import Map
from neuromancer import blocks
from neuromancer.integers import IntegerProjection, IntegerInequalityProjection


def arg_mpLP_problem(prefix=''):
    """
    Command line parser for mpLP problem definition arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("mpLP")
    gp.add("-Q", type=float, default=1.0,
           help="loss function weight.")  # tuned value: 1.0
    gp.add("-Q_sub", type=float, default=0.0,
           help="regularization weight.")
    gp.add("-Q_con", type=float, default=100.0,
           help="constraints penalty weight.")  # tuned value: 1.0
    gp.add("-nx_hidden", type=int, default=80,
           help="Number of hidden states of the solution map")
    gp.add("-n_layers", type=int, default=4,
           help="Number of hidden layers of the solution map")
    gp.add("-bias", action="store_true",
           help="Whether to use bias in the neural network block component models.")
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    gp.add("-epochs", type=int, default=1000,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.001,
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=100,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=100,
           help="Number of epochs to wait before enacting early stopping policy.")
    gp.add("-loss", type=str, default='penalty',
           choices=['penalty', 'augmented_lagrange', 'barrier'],
           help="type of the loss function.")
    gp.add("-barrier_type", type=str, default='log10',
           choices=['log', 'log10', 'inverse'],
           help="type of the barrier function in the barrier loss.")
    gp.add("-eta", type=float, default=0.99,
           help="eta in augmented lagrangian.")
    gp.add("-sigma", type=float, default=2.0,
           help="sigma in augmented lagrangian.")
    gp.add("-mu_init", type=float, default=1.,
           help="mu_init in augmented lagrangian.")
    gp.add("-mu_max", type=float, default=1000.,
           help="mu_max in augmented lagrangian.")
    gp.add("-inner_loop", type=int, default=1,
           help="inner loop in augmented lagrangian")
    gp.add("-train_integer", default=True, choices=[True, False],
           help="Whether to use integer update during training or not.")
    gp.add("-inference_integer", default=False, choices=[True, False],
           help="Whether to use integer update during inference or not.")
    gp.add("-train_proj_int_ineq", default=False, choices=[True, False],
           help="Whether to use integer constraints projection during training or not.")
    gp.add("-inference_proj_int_ineq", default=True, choices=[True, False],
           help="Whether to use integer constraints projection during inference or not.")
    gp.add("-n_projections_train", type=int, default=1,
           help="number of mip constraints projection steps during training")
    gp.add("-n_projections_inference", type=int, default=10,
           help="number of mip constraints projections steps at the inference time")
    gp.add("-proj_dropout", type=float, default=0.5,
           help="random dropout of the mip constraints projections.")
    gp.add("-direction", default='gradient',
           choices=['gradient', 'random'],
           help="method for obtaining directions for integer constraints projections.")
    return parser


if __name__ == "__main__":
    """
    # # #  optimization problem hyperparameters
    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_mpLP_problem()])
    args, grps = parser.parse_arg_groups()
    args.bias = True
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    """
    # # #  Dataset 
    """
    #  randomly sampled parameters theta generating superset of:
    #  theta_samples.min() <= theta <= theta_samples.max()
    np.random.seed(args.data_seed)
    nsim = 9000  # number of datapoints: increase sample density for more robust results
    samples = {"p1": np.random.uniform(low=1.0, high=11.0, size=(nsim, 1)),
               "p2": np.random.uniform(low=1.0, high=11.0, size=(nsim, 1))}
    data, dims = get_static_dataloaders(samples)
    train_data, dev_data, test_data = data

    """
    # # #  mpQP primal solution map architecture
    """
    func = blocks.MLP(insize=2, outsize=2,
                    bias=True,
                    linear_map=slim.maps['linear'],
                    nonlin=activations['relu'],
                    hsizes=[args.nx_hidden] * args.n_layers)
    sol_map = Map(func,
            input_keys=["p1", "p2"],
            output_keys=["x"],
            name='primal_map')

    """
    # # #  mpQP objective and constraints formulation in Neuromancer
    """
    # variables
    x = variable("x")[:, [0]]
    y = variable("x")[:, [1]]
    # sampled parameters
    p1 = variable('p1')
    p2 = variable('p2')

    # objective function
    f = x + 2*y
    obj = f.minimize(weight=args.Q, name='obj')
    # constraints
    g1 = -x - y + p1
    con_1 = (g1 <= 0)
    con_1.name = 'c1'
    g2 = x + y - p1 - 5
    con_2 = (g2 <= 0)
    con_2.name = 'c2'
    g3 = x - y + p2 - 5
    con_3 = (g3 <= 0)
    con_3.name = 'c3'
    g4 = -x + y - p2
    con_4 = (g4 <= 0)
    con_4.name = 'c4'

    """
    # # #  mpMILP problem formulation in Neuromancer
    """
    # list of objectives, constraints, and components (solution maps)
    objectives = [obj]
    constraints = [args.Q_con*con_1, args.Q_con*con_2,
                   args.Q_con*con_3, args.Q_con*con_4]
    components = [sol_map]

    if args.train_integer:  # MIQP = use integer correction update during training
        integer_map = IntegerProjection(input_keys=['x'],
                                        method='round_sawtooth',
                                        nsteps=1, stepsize=0.2,
                                        name='int_map')
        components.append(integer_map)
    if args.train_proj_int_ineq:
        int_projection = IntegerInequalityProjection(constraints, input_keys=["x"],
                                                     n_projections=args.n_projections_train,
                                                     dropout=args.proj_dropout,
                                                     direction=args.direction,
                                                     nsteps=3, stepsize=0.1, name='proj_int')
        components.append(int_projection)

    # create constrained optimization loss
    loss = get_loss(objectives, constraints, train_data, args)
    # construct constrained optimization problem
    problem = Problem(components, loss, grad_inference=args.train_proj_int_ineq)
    # plot computational graph
    problem.plot_graph()

    """
    # # # Metrics and Logger
    """
    args.savedir = 'test_mpMILP'
    args.verbosity = 1
    metrics = ["train_loss", "train_obj", "train_mu_scaled_penalty_loss", "train_con_lagrangian",
               "train_mu", "train_c1"]
    if args.logger == 'stdout':
        Logger = BasicLogger
    elif args.logger == 'mlflow':
        Logger = MLFlowLogger
    logger = Logger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'mpMILP'


    """
    # # #  mpMILP problem solution in Neuromancer
    """
    optimizer = torch.optim.AdamW(problem.parameters(), lr=args.lr)

    # define trainer
    trainer = Trainer(
        problem,
        train_data,
        dev_data,
        test_data,
        optimizer,
        logger=logger,
        epochs=args.epochs,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
        patience=args.patience,
        warmup=args.warmup,
        device=device,
    )

    # Train mpLP solution map
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)
    # load best model dict
    problem.load_state_dict(best_model)

    """
    CVXPY benchmark
    """
    # Define and solve the CVXPY problem.
    p1 = 10.0  # problem parameter

    def MILP_param(p1, p2):
        x = cp.Variable(1, integer=True)
        y = cp.Variable(1, integer=True)
        prob = cp.Problem(cp.Minimize(x + 2*y),
                          [-x - y + p1 <= 0,
                           x + y - p1 - 5 <= 0,
                           x - y + p2 - 5 <= 0,
                           -x + y - p2 <= 0])
        return prob, x, y

    """
    MIP Integer correction at inference
    """
    # integer projection to nearest integer
    int_map = IntegerProjection(input_keys=['x'],
                                   method='round_sawtooth',
                                   nsteps=1, stepsize=1.0,
                                   name='int_map')
    if args.inference_integer:
        if args.train_integer:
            problem.components[1] = int_map
        else:
            problem.components.append(int_map)

    # integer projection to feasible set
    int_projection = IntegerInequalityProjection(constraints, input_keys=["x"],  method="sawtooth",
                                                 n_projections=args.n_projections_inference,
                                                 dropout=args.proj_dropout,
                                                 direction=args.direction,
                                                 nsteps=1, stepsize=1.0, name='proj_int')
    if args.inference_proj_int_ineq:
        if args.train_proj_int_ineq:
            if args.train_integer:
                problem.components[2] = int_projection
            else:
                problem.components[1] = int_projection
        else:
            problem.components.append(int_projection)

    """
    Plots
    """
    # # # single plot  # # #
    plt.rc('axes', titlesize=14)  # fontsize of the title
    plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=14)  # fontsize of the y tick labels

    p = 3.0
    x1 = np.arange(-2.0, 10.0, 0.05)
    y1 = np.arange(-2.0, 10.0, 0.05)
    xx, yy = np.meshgrid(x1, y1)

    # eval objective and constraints
    J = xx + 2 * yy
    c1 = xx + yy - p
    c2 = -xx - yy + p + 5
    c3 = -xx + yy - p + 5
    c4 = xx - yy + p

    fig, ax = plt.subplots(1, 1)
    # Plot constraints and loss contours
    levels = [-6, -3, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    cp_plot = ax.contour(xx, yy, J, levels=levels, alpha=0.4)
    cp_plot = ax.contourf(xx, yy, J, levels=levels, alpha=0.4)
    cg1 = ax.contour(xx, yy, c1, [0], colors='mediumblue', alpha=0.7)
    cg2 = ax.contour(xx, yy, c2, [0], colors='mediumblue', alpha=0.7)
    cg3 = ax.contour(xx, yy, c3, [0], colors='mediumblue', alpha=0.7)
    cg4 = ax.contour(xx, yy, c4, [0], colors='mediumblue', alpha=0.7)
    plt.setp(cg1.collections,
             path_effects=[patheffects.withTickedStroke()], alpha=0.7)
    plt.setp(cg2.collections,
             path_effects=[patheffects.withTickedStroke()], alpha=0.7)
    plt.setp(cg3.collections,
             path_effects=[patheffects.withTickedStroke()], alpha=0.7)
    plt.setp(cg4.collections,
             path_effects=[patheffects.withTickedStroke()], alpha=0.7)
    fig.colorbar(cp_plot)

    # Solve MIQP via neuromancer
    datapoint = {}
    datapoint['p1'] = torch.tensor([[p]])
    datapoint['p2'] = torch.tensor([[p]])
    datapoint['name'] = "test"
    model_out = problem(datapoint)

    # intermediate solutions
    X = []
    Y = []
    if args.inference_proj_int_ineq:
        for k in range(args.n_projections_inference+2):
            x_nm_k = model_out['test_' + "x" + f'_{k}'][0, 0].detach().numpy()
            y_nm_k = model_out['test_' + "x" + f'_{k}'][0, 1].detach().numpy()
            X.append(x_nm_k)
            Y.append(y_nm_k)
            marker_size = 5 + k * (10 / args.n_projections_inference+2)
            ax.plot(x_nm_k, y_nm_k, 'g*', markersize=marker_size)
        ax.plot(np.asarray(X), np.asarray(Y), 'g--')

    # final solution
    x_nm = model_out['test_' + "x"][0, 0].detach().numpy()
    y_nm = model_out['test_' + "x"][0, 1].detach().numpy()
    print(x_nm)
    print(y_nm)
    ax.plot(x_nm, y_nm, 'r*', markersize=20)

    # Plot admissible integer solutions
    x_int = np.arange(-2.0, 10.0, 1.0)
    y_int = np.arange(-2.0, 10.0, 1.0)
    xx, yy = np.meshgrid(x_int, y_int)
    ax.plot(xx, yy, 'bo', markersize=3.5)
    ax.set_ylim(-2.0, 8.0)
    ax.set_xlim(-2.0, 8.0)
    fig.tight_layout()

    print(f'primal solution MIQP x={x.value}, y={y.value}')
    print(f'parameter p={p}')
    print(f'primal solution DPP x1={x_nm}, x2={y_nm}')
    print(f' f: {model_out["test_" + f.key]}')
    print(f' g1: {model_out["test_" + g1.key]}')

    # # #  test set of problem parameters  # # #
    plt.rc('axes', titlesize=10)  # fontsize of the title
    plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=10)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=10)  # fontsize of the y tick labels

    params = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    x1 = np.arange(-1.0, 10.0, 0.05)
    y1 = np.arange(-1.0, 10.0, 0.05)
    xx, yy = np.meshgrid(x1, y1)
    fig, ax = plt.subplots(3, 3)
    row_id = 0
    column_id = 0
    for i, p in enumerate(params):
        if i % 3 == 0 and i != 0:
            row_id += 1
            column_id = 0
        # eval objective and constraints
        J = xx + 2*yy
        c1 = xx + yy - p
        c2 = -xx - yy + p + 5
        c3 = -xx + yy - p + 5
        c4 = xx - yy + p

        # Plot constraints and loss contours
        cp_plot = ax[row_id, column_id].contourf(xx, yy, J, 50, alpha=0.4)
        ax[row_id, column_id].set_title(f'MILP p={p}')
        cg1 = ax[row_id, column_id].contour(xx, yy, c1, [0], colors='mediumblue', alpha=0.7)
        cg2 = ax[row_id, column_id].contour(xx, yy, c2, [0], colors='mediumblue', alpha=0.7)
        cg3 = ax[row_id, column_id].contour(xx, yy, c3, [0], colors='mediumblue', alpha=0.7)
        cg4 = ax[row_id, column_id].contour(xx, yy, c4, [0], colors='mediumblue', alpha=0.7)
        plt.setp(cg1.collections,
                 path_effects=[patheffects.withTickedStroke()], alpha=0.7)
        plt.setp(cg2.collections,
                 path_effects=[patheffects.withTickedStroke()], alpha=0.7)
        plt.setp(cg3.collections,
                 path_effects=[patheffects.withTickedStroke()], alpha=0.7)
        plt.setp(cg4.collections,
                 path_effects=[patheffects.withTickedStroke()], alpha=0.7)
        fig.colorbar(cp_plot, ax=ax[row_id, column_id])

        # Solve LP
        prob, x, y = MILP_param(p, p)
        prob.solve(solver='ECOS_BB', verbose=True)

        # Solve MILP via neuromancer
        datapoint = {}
        datapoint['p1'] = torch.tensor([[p]])
        datapoint['p2'] = torch.tensor([[p]])
        datapoint['name'] = "test"
        model_out = problem(datapoint)
        x_nm = model_out['test_' + "x"][0, 0].detach().numpy()
        y_nm = model_out['test_' + "x"][0, 1].detach().numpy()

        print(f'primal solution MILP x={x.value}, y={y.value}')
        print(f'parameter p={p}')
        print(f'primal solution DPP x1={x_nm}, x2={y_nm}')
        print(f' f: {model_out["test_" + f.key]}')
        print(f' g1: {model_out["test_" + g1.key]}')

        # Plot optimal solutions
        ax[row_id, column_id].plot(x.value, y.value, 'g*', markersize=10)
        ax[row_id, column_id].plot(x_nm, y_nm, 'r*', markersize=10)
        # Plot admissible integer solutions
        x_int = np.arange(-1.0, 10.0, 1.0)
        y_int = np.arange(-1.0, 10.0, 1.0)
        xx, yy = np.meshgrid(x_int, y_int)
        ax[row_id, column_id].plot(xx, yy, 'bo', markersize=2)

        column_id += 1
    plt.show()
    plt.savefig('figs/mpMILP_nm.png')


    """
    Benchmark Solution
    """

    def eval_constraints(x, y, p1, p2):
        """
        evaluate mean constraints violations
        """
        con_1_viol = np.maximum(0, -x - y + p1)
        con_2_viol = np.maximum(0, x + y - p1 - 5)
        con_3_viol = np.maximum(0, x - y + p2 - 5)
        con_4_viol = np.maximum(0, -x + y - p2)
        con_viol = con_1_viol + con_2_viol + con_3_viol + con_4_viol
        con_viol_mean = np.mean(con_viol)
        return con_viol_mean

    def eval_objective(x, y, a1=1, a2=2):
        obj_value_mean = np.mean(a1 * x + a2 * y)
        return obj_value_mean

    # fix random seeds
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)

    # select n number of random samples to evaluate
    n_samples = 10
    idx = np.random.randint(0, nsim, n_samples)
    p1 = samples['p1'][idx]
    p2 = samples['p2'][idx]

    # create named dictionary for neuromancer
    datapoint = {}
    datapoint['p1'] = torch.tensor(p1).float()
    datapoint['p2'] = torch.tensor(p2).float()
    datapoint['name'] = "test"

    # Solve via neuromancer
    t = time.time()
    model_out = problem(datapoint)
    nm_time = time.time() - t
    x_nm = model_out['test_' + "x"][:, [0]].detach().numpy()
    y_nm = model_out['test_' + "x"][:, [1]].detach().numpy()

    # Solve via solver
    t = time.time()
    x_solver, y_solver = [], []
    for i in range(0, n_samples):
        prob, x, y = MILP_param(p1[i], p2[i])
        prob.solve(solver='ECOS_BB', verbose=True)
        prob.solve()
        x_solver.append(x.value)
        y_solver.append(y.value)
    solver_time = time.time() - t
    x_solver = np.asarray(x_solver)
    y_solver = np.asarray(y_solver)

    # Evaluate neuromancer solution
    print(f'Solution for {n_samples} problems via Neuromancer obtained in {nm_time:.4f} seconds')
    nm_con_viol_mean = eval_constraints(x_nm, y_nm, p1, p2)
    print(f'Neuromancer mean constraints violation {nm_con_viol_mean:.4f}')
    nm_obj_mean = eval_objective(x_nm, y_nm)
    print(f'Neuromancer mean objective value {nm_obj_mean:.4f}')

    # Evaluate solver solution
    print(f'Solution for {n_samples} problems via solver obtained in {solver_time:.4f} seconds')
    solver_con_viol_mean = eval_constraints(x_solver, y_solver, p1, p2)
    print(f'Solver mean constraints violation {solver_con_viol_mean:.4f}')
    solver_obj_mean = eval_objective(x_solver, y_solver)
    print(f'Solver mean objective value {solver_obj_mean:.4f}')

    # neuromancer solver comparison
    speedup_factor = solver_time/nm_time
    print(f'Solution speedup factor {speedup_factor:.4f}')

    # Difference in primal optimizers
    dx = (x_solver - x_nm)[:,0]
    dy = (y_solver - y_nm)[:,0]
    err_x = np.mean(dx**2)
    err_y = np.mean(dy**2)
    err_primal = err_x + err_y
    print('MSE primal optimizers:', err_primal)

    # Difference in objective
    err_obj = np.abs(solver_obj_mean - nm_obj_mean) / solver_obj_mean * 100
    print(f'mean objective value discrepancy: {err_obj:.2f} %')

    # stats to log
    stats = {"n_samples": n_samples,
             "nm_time": nm_time,
             "nm_con_viol_mean": nm_con_viol_mean,
             "nm_obj_mean": nm_obj_mean,
             "solver_time": solver_time,
             "solver_con_viol_mean": solver_con_viol_mean,
             "solver_obj_mean": solver_obj_mean,
             "speedup_factor": speedup_factor,
             "err_primal": err_primal,
             "err_obj": err_obj}