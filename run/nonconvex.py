#!/usr/bin/env python
# coding: utf-8
"""
Experiment pipeline for QP
"""
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from run import utils

# turn off warning
import logging
logging.getLogger("pyomo.core").setLevel(logging.ERROR)

def exact(loader_test, config):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"EX in NC for size {config.size}.")
    # config
    num_var = config.size
    num_eq = config.size // 10
    num_ineq = config.size // 2
    # init model
    from src.problem import msNonconvex
    model = msNonconvex(num_var, num_eq, num_ineq, timelimit=1000)
    # init df
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # go through test data
    b_test = loader_test.dataset.datadict["b"][:100]
    for b in tqdm(list(b_test)):
        # set params
        model.set_param_val({"b":b.cpu().numpy()})
        # solve
        tick = time.time()
        params.append(list(b.cpu().numpy()))
        try:
            xval, objval = model.solve("scip")
            tock = time.time()
            # eval
            sols.append(list(list(xval.values())[0].values()))
            objvals.append(objval)
            viol = model.cal_violation()
            mean_viols.append(np.mean(viol))
            max_viols.append(np.max(viol))
            num_viols.append(np.sum(viol > 1e-6))
        except:
            # infeasible
            sols.append(None)
            objvals.append(None)
            mean_viols.append(None)
            max_viols.append(None)
            num_viols.append(None)
            tock = time.time()
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param": params,
                       "Sol": sols,
                       "Obj Val": objvals,
                       "Mean Violation": mean_viols,
                       "Max Violation": max_viols,
                       "Num Violations": num_viols,
                       "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    print("Number of unsolved instances: ", df["Sol"].isna().sum())
    df.to_csv(f"result/nc_exact_{num_var}-{num_ineq}.csv")


def relRnd(loader_test, config):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RR in NC for size {config.size}.")
    from src.heuristic import naive_round
    # config
    num_var = config.size
    num_eq = config.size // 10
    num_ineq = config.size // 2
    # init model
    from src.problem import msNonconvex
    model = msNonconvex(num_var, num_eq, num_ineq, timelimit=1000)
    # init df
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # go through test data
    b_test = loader_test.dataset.datadict["b"][:100]
    for b in tqdm(list(b_test)):
        # set params
        model.set_param_val({"b":b.cpu().numpy()})
        # relax
        model_rel = model.relax()
        # solve
        tick = time.time()
        params.append(list(b.cpu().numpy()))
        try:
            xval_rel, _ = model_rel.solve("scip")
            xval, objval = naive_round(xval_rel, model)
            tock = time.time()
            # eval
            sols.append(list(list(xval.values())[0].values()))
            objvals.append(objval)
            viol = model.cal_violation()
            mean_viols.append(np.mean(viol))
            max_viols.append(np.max(viol))
            num_viols.append(np.sum(viol > 1e-6))
        except:
            # infeasible
            sols.append(None)
            objvals.append(None)
            mean_viols.append(None)
            max_viols.append(None)
            num_viols.append(None)
            tock = time.time()
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param": params,
                       "Sol": sols,
                       "Obj Val": objvals,
                       "Mean Violation": mean_viols,
                       "Max Violation": max_viols,
                       "Num Violations": num_viols,
                       "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    print("Number of unsolved instances: ", df["Sol"].isna().sum())
    df.to_csv(f"result/nc_rel_{num_var}-{num_ineq}.csv")


def root(loader_test, config):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"N1 in NC for size {config.size}.")
    # config
    num_var = config.size
    num_eq = config.size // 10
    num_ineq = config.size // 2
    # init model
    from src.problem import msNonconvex
    model = msNonconvex(num_var, num_eq, num_ineq, timelimit=1000)
    model_heur = model.first_solution_heuristic(nodes_limit=1)
    # init df
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    # go through test data
    b_test = loader_test.dataset.datadict["b"][:100]
    for b in tqdm(list(b_test)):
        # set params
        model_heur.set_param_val({"b":b.cpu().numpy()})
        # solve
        tick = time.time()
        params.append(list(b.cpu().numpy()))
        try:
            xval, objval = model_heur.solve("scip")
            tock = time.time()
            # eval
            sols.append(list(list(xval.values())[0].values()))
            objvals.append(objval)
            viol = model_heur.cal_violation()
            mean_viols.append(np.mean(viol))
            max_viols.append(np.max(viol))
            num_viols.append(np.sum(viol > 1e-6))
        except:
            # infeasible
            sols.append(None)
            objvals.append(None)
            mean_viols.append(None)
            max_viols.append(None)
            num_viols.append(None)
            tock = time.time()
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param": params,
                       "Sol": sols,
                       "Obj Val": objvals,
                       "Mean Violation": mean_viols,
                       "Max Violation": max_viols,
                       "Num Violations": num_viols,
                       "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    print("Number of unsolved instances: ", df["Sol"].isna().sum())
    df.to_csv(f"result/nc_root_{num_var}-{num_ineq}.csv")


def rndCls(loader_train, loader_test, loader_val, config):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RC in NC for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmNonconvex
    from src.func.layer import netFC
    from src.func import roundGumbelModel, completePartial
    # config
    num_var = config.size
    num_eq = config.size // 10
    num_ineq = config.size // 2
    hlayers_sol = config.hlayers_sol
    hlayers_rnd = config.hlayers_rnd
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    # init model
    from src.problem import msNonconvex
    model = msNonconvex(num_var, num_eq, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = netFC(input_dim=num_eq, hidden_dims=[hsize]*hlayers_sol, output_dim=num_var-num_eq)
    smap = nm.system.Node(func, ["b"], ["x"], name="smap")
    # define rounding model
    layers_rnd = netFC(input_dim=num_var, hidden_dims=[hsize]*hlayers_rnd,
                       output_dim=num_var-num_eq)
    rnd = roundGumbelModel(layers=layers_rnd, param_keys=["b"], var_keys=["x"],
                           output_keys=["x_rnd"], int_ind=model.int_ind,
                           continuous_update=True, name="round")
    # fill variables from linear system
    complete = completePartial(A=torch.from_numpy(model.A).float().cuda(), num_var=num_var,
                               partial_ind=range(num_var-num_eq), var_key="x_rnd",
                               rhs_key="b", output_key="x_comp", name="Complete")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd, complete]).to("cuda")
    loss_fn = nmNonconvex.penaltyLoss(["b", "x_comp"], num_var, num_eq, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr)
    # eval
    df = eval(components, model, loader_test)
    df.to_csv(f"result/nc_cls{penalty_weight}_{num_var}-{num_ineq}.csv")


def rndThd(loader_train, loader_test, loader_val, config):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"LT in NC for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmNonconvex
    from src.func.layer import netFC
    from src.func import thresholdModel, completePartial
    # config
    num_var = config.size
    num_eq = config.size // 10
    num_ineq = config.size // 2
    hlayers_sol = config.hlayers_sol
    hlayers_rnd = config.hlayers_rnd
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    # init model
    from src.problem import msNonconvex
    model = msNonconvex(num_var, num_eq, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = netFC(input_dim=num_eq, hidden_dims=[hsize]*hlayers_sol, output_dim=num_var-num_eq)
    smap = nm.system.Node(func, ["b"], ["x"], name="smap")
    # define rounding model
    layers_rnd = netFC(input_dim=num_var, hidden_dims=[hsize]*hlayers_rnd,
                       output_dim=num_var-num_eq)
    rnd = thresholdModel(layers=layers_rnd, param_keys=["b"], var_keys=["x"],
                         output_keys=["x_rnd"], int_ind=model.int_ind,
                         continuous_update=True, name="round")
    # fill variables from linear system
    complete = completePartial(A=torch.from_numpy(model.A).float().cuda(), num_var=num_var,
                               partial_ind=range(num_var-num_eq), var_key="x_rnd",
                               rhs_key="b", output_key="x_comp", name="Complete")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd, complete]).to("cuda")
    loss_fn = nmNonconvex.penaltyLoss(["b", "x_comp"], num_var, num_eq, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr)
    # eval
    df = eval(components, model, loader_test)
    df.to_csv(f"result/nc_thd{penalty_weight}_{num_var}-{num_ineq}.csv")


def lrnRnd(loader_train, loader_test, loader_val, config):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RL in CQ for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmNonconvex
    from src.func.layer import netFC
    from src.func import completePartial
    # config
    num_var = config.size
    num_eq = config.size // 10
    num_ineq = config.size // 2
    hlayers_sol = config.hlayers_sol
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    # init model
    from src.problem import msNonconvex
    model = msNonconvex(num_var, num_eq, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = netFC(input_dim=num_eq, hidden_dims=[hsize]*hlayers_sol, output_dim=num_var-num_eq)
    smap = nm.system.Node(func, ["b"], ["x"], name="smap")
    # fill variables from linear system
    complete = completePartial(A=torch.from_numpy(model.A).float().cuda(), num_var=num_var,
                               partial_ind=range(num_var-num_eq), var_key="x",
                               rhs_key="b", output_key="x_comp", name="Complete")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, complete]).to("cuda")
    loss_fn = nmNonconvex.penaltyLoss(["b", "x_comp"], num_var, num_eq, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr)
    # eval
    from src.heuristic import naive_round
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    for b in tqdm(loader_test.dataset.datadict["b"][:100]):
        # data point as tensor
        datapoints = {"b": torch.unsqueeze(b, 0).to("cuda"),
                      "name": "test"}
        # infer
        components.eval()
        tick = time.time()
        with torch.no_grad():
            for comp in components:
                datapoints.update(comp(datapoints))
        tock = time.time()
        # assign params
        model.set_param_val({"b":b.cpu().numpy()})
        # assign vars
        x = datapoints["x_comp"]
        for i in range(num_var):
            model.vars["x"][i].value = x[0,i].item()
        # get solutions
        xval_rel, _ = model.get_val()
        xval, objval = naive_round(xval_rel, model)
        params.append(list(b.cpu().numpy()))
        sols.append(list(list(xval.values())[0].values()))
        objvals.append(objval)
        viol = model.cal_violation()
        mean_viols.append(np.mean(viol))
        max_viols.append(np.max(viol))
        num_viols.append(np.sum(viol > 1e-6))
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param": params,
                       "Sol": sols,
                       "Obj Val": objvals,
                       "Mean Violation": mean_viols,
                       "Max Violation": max_viols,
                       "Num Violations": num_viols,
                       "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    df.to_csv(f"result/nc_lrn{penalty_weight}_{num_var}-{num_ineq}.csv")


def rndSte(loader_train, loader_test, loader_val, config):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RS in NC for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmNonconvex
    from src.func.layer import netFC
    from src.func import roundSTEModel, completePartial
    # config
    num_var = config.size
    num_eq = config.size // 10
    num_ineq = config.size // 2
    hlayers_sol = config.hlayers_sol
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    # init model
    from src.problem import msNonconvex
    model = msNonconvex(num_var, num_eq, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = netFC(input_dim=num_eq, hidden_dims=[hsize]*hlayers_sol, output_dim=num_var-num_eq)
    smap = nm.system.Node(func, ["b"], ["x"], name="smap")
    # define rounding model
    rnd = roundSTEModel(param_keys=["b"], var_keys=["x"], output_keys=["x_rnd"], int_ind=model.int_ind, name="round")
    # fill variables from linear system
    complete = completePartial(A=torch.from_numpy(model.A).float().cuda(), num_var=num_var,
                               partial_ind=range(num_var-num_eq), var_key="x_rnd",
                               rhs_key="b", output_key="x_comp", name="Complete")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd, complete]).to("cuda")
    loss_fn = nmNonconvex.penaltyLoss(["b", "x_comp"], num_var, num_eq, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr)
    # eval
    df = eval(components, model, loader_test)
    df.to_csv(f"result/nc_ste{penalty_weight}_{num_var}-{num_ineq}.csv")


def eval(components, model, loader_test):
    params, sols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], []
    b_test = loader_test.dataset.datadict["b"][:100]
    for b in tqdm(list(b_test)):
        # data point as tensor
        datapoints = {"b": torch.unsqueeze(b, 0).to("cuda"),
                      "name": "test"}
        # infer
        components.eval()
        tick = time.time()
        with torch.no_grad():
            for comp in components:
                datapoints.update(comp(datapoints))
        tock = time.time()
        # assign params
        model.set_param_val({"b":b.cpu().numpy()})
        # assign vars
        x = datapoints["x_comp"]
        for i in range(len(model.vars["x"])):
            model.vars["x"][i].value = x[0,i].item()
        # get solutions
        xval, objval = model.get_val()
        params.append(list(b.cpu().numpy()))
        sols.append(list(list(xval.values())[0].values()))
        objvals.append(objval)
        viol = model.cal_violation()
        mean_viols.append(np.mean(viol))
        max_viols.append(np.max(viol))
        num_viols.append(np.sum(viol > 1e-6))
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param": params,
                       "Sol": sols,
                       "Obj Val": objvals,
                       "Mean Violation": mean_viols,
                       "Max Violation": max_viols,
                       "Num Violations": num_viols,
                       "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    return df
