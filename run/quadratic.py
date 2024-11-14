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
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"Ex for size {config.size}.")
    # config
    num_var = config.size
    num_ineq = config.size
    # init model
    from src.problem import msQuadratic
    model = msQuadratic(num_var, num_ineq, timelimit=1000)
    # init df
    params, sols, objvals, conviols, elapseds = [], [], [], [], []
    # go through test data
    for b in tqdm(loader_test.dataset.datadict["b"][:100]):
        # set params
        model.set_param_val({"b":b.cpu().numpy()})
        # solve
        tick = time.time()
        try:
            xval, objval = model.solve("gurobi")
            # eval
            params.append(list(b.cpu().numpy()))
            sols.append(list(list(xval.values())[0].values()))
            objvals.append(objval)
            conviols.append(sum(model.cal_violation()))
        except:
            # infeasible
            params.append(list(b.cpu().numpy()))
            sols.append(None)
            objvals.append(None)
            conviols.append(None)
        tock = time.time()
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param":params,
                       "Sol":sols,
                       "Obj Val": objvals,
                       "Constraints Viol": conviols,
                       "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solution: {}".format(np.sum(df["Constraints Viol"] > 0)))
    print("Number of unsolved instances: ", df["Sol"].isna().sum())
    df.to_csv(f"result/cq_exact_{num_var}-{num_ineq}.csv")


def relRnd(loader_test, config):
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RR for size {config.size}.")
    from src.heuristic import naive_round
    # config
    num_var = config.size
    num_ineq = config.size
    # init model
    from src.problem import msQuadratic
    model = msQuadratic(num_var, num_ineq, timelimit=1000)
    # init df
    params, sols, objvals, conviols, elapseds = [], [], [], [], []
    # go through test data
    for b in tqdm(loader_test.dataset.datadict["b"][:100]):
        # set params
        model.set_param_val({"b":b.cpu().numpy()})
        # relax
        model_rel = model.relax()
        # solve
        tick = time.time()
        try:
            xval_rel, _ = model_rel.solve("gurobi")
            xval, objval = naive_round(xval_rel, model)
            # eval
            params.append(list(b.cpu().numpy()))
            sols.append(list(list(xval.values())[0].values()))
            objvals.append(objval)
            conviols.append(sum(model.cal_violation()))
        except:
            # infeasible
            params.append(list(b.cpu().numpy()))
            sols.append(None)
            objvals.append(None)
            conviols.append(None)
        tock = time.time()
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param":params,
                       "Sol":sols,
                       "Obj Val": objvals,
                       "Constraints Viol": conviols,
                       "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Constraints Viol"] > 0)))
    print("Number of unsolved instances: ", df["Sol"].isna().sum())
    df.to_csv(f"result/cq_rel_{num_var}-{num_ineq}.csv")


def root(loader_test, config):
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"N1 for size {config.size}.")
    # config
    num_var = config.size
    num_ineq = config.size
    # init model
    from src.problem import msQuadratic
    model = msQuadratic(num_var, num_ineq, timelimit=1000)
    model_heur = model.first_solution_heuristic(nodes_limit=1)
    # init df
    params, sols, objvals, conviols, elapseds = [], [], [], [], []
    # go through test data
    for b in tqdm(loader_test.dataset.datadict["b"][:100]):
        # set params
        model_heur.set_param_val({"b":b.cpu().numpy()})
        # solve
        tick = time.time()
        try:
            xval, objval = model_heur.solve("gurobi")
            # eval
            params.append(list(b.cpu().numpy()))
            sols.append(list(list(xval.values())[0].values()))
            objvals.append(objval)
            conviols.append(sum(model_heur.cal_violation()))
        except:
            # infeasible
            params.append(list(b.cpu().numpy()))
            sols.append(None)
            objvals.append(None)
            conviols.append(None)
        tock = time.time()
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param":params,
                       "Sol":sols,
                       "Obj Val": objvals,
                       "Constraints Viol": conviols,
                       "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solution: {}".format(np.sum(df["Constraints Viol"] > 0)))
    print("Number of unsolved instances: ", df["Sol"].isna().sum())
    df.to_csv(f"result/cq_root_{num_var}-{num_ineq}.csv")


def rndCls(loader_train, loader_test, loader_val, config, penalty_growth=False):
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RC for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmQuadratic
    from src.func.layer import netFC
    from src.func import roundGumbelModel
    # config
    num_var = config.size
    num_ineq = config.size
    hlayers_sol = config.hlayers_sol
    hlayers_rnd = config.hlayers_rnd
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    # init model
    from src.problem import msQuadratic
    model = msQuadratic(num_var, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_ineq, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["b"], ["x"], name="smap")
    # define rounding model
    layers_rnd = netFC(input_dim=num_ineq+num_var,
                       hidden_dims=[hsize]*hlayers_rnd,
                       output_dim=num_var)
    rnd = roundGumbelModel(layers=layers_rnd, param_keys=["b"], var_keys=["x"],
                           output_keys=["x_rnd"], int_ind=model.int_ind,
                           continuous_update=True, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmQuadratic(["b", "x_rnd"], num_var, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    df = eval(components, model, loader_test)
    if penalty_growth:
        df.to_csv(f"result/cq_cls{penalty_weight}_{num_var}-{num_ineq}-g.csv")
    else:
        df.to_csv(f"result/cq_cls{penalty_weight}_{num_var}-{num_ineq}.csv")


def rndThd(loader_train, loader_test, loader_val, config, penalty_growth=False):
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"LT for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmQuadratic
    from src.func.layer import netFC
    from src.func import roundThresholdModel
    # config
    num_var = config.size
    num_ineq = config.size
    hlayers_sol = config.hlayers_sol
    hlayers_rnd = config.hlayers_rnd
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    # init model
    from src.problem import msQuadratic
    model = msQuadratic(num_var, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_ineq, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["b"], ["x"], name="smap")
    # define rounding model
    layers_rnd = netFC(input_dim=num_ineq+num_var,
                       hidden_dims=[hsize]*hlayers_rnd,
                       output_dim=num_var)
    rnd = roundThresholdModel(layers=layers_rnd, param_keys=["b"], var_keys=["x"],
                              output_keys=["x_rnd"], int_ind=model.int_ind,
                              continuous_update=True, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmQuadratic(["b", "x_rnd"], num_var, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    df = eval(components, model, loader_test)
    if penalty_growth:
        df.to_csv(f"result/cq_thd{penalty_weight}_{num_var}-{num_ineq}-g.csv")
    else:
        df.to_csv(f"result/cq_thd{penalty_weight}_{num_var}-{num_ineq}.csv")


def lrnRnd(loader_train, loader_test, loader_val, config, penalty_growth=False):
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RL for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmQuadratic
    from src.func.layer import netFC
    from src.func import roundThresholdModel
    # config
    num_var = config.size
    num_ineq = config.size
    hlayers_sol = config.hlayers_sol
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    # init model
    from src.problem import msQuadratic
    model = msQuadratic(num_var, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_ineq, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["b"], ["x"], name="smap")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap]).to("cuda")
    loss_fn = nmQuadratic(["b", "x"], num_var, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    from src.heuristic import naive_round
    params, sols, objvals, conviols, elapseds = [], [], [], [], []
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
        x = datapoints["x"]
        for i in range(num_var):
            model.vars["x"][i].value = x[0,i].item()
        # get solutions
        xval_rel, _ = model.get_val()
        xval, objval = naive_round(xval_rel, model)
        params.append(list(b.cpu().numpy()))
        sols.append(list(list(xval.values())[0].values()))
        objvals.append(objval)
        conviols.append(sum(model.cal_violation()))
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param":params,
                       "Sol":sols,
                       "Obj Val": objvals,
                       "Constraints Viol": conviols,
                       "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solution: {}".format(np.sum(df["Constraints Viol"] > 0)))
    if penalty_growth:
        df.to_csv(f"result/cq_lrn{penalty_weight}_{num_var}-{num_ineq}-g.csv")
    else:
        df.to_csv(f"result/cq_lrn{penalty_weight}_{num_var}-{num_ineq}.csv")


def rndSte(loader_train, loader_test, loader_val, config, penalty_growth=False):
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RS for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmQuadratic
    from src.func.layer import netFC
    from src.func import roundSTEModel
    # config
    num_var = config.size
    num_ineq = config.size
    hlayers_sol = config.hlayers_sol
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    # init model
    from src.problem import msQuadratic
    model = msQuadratic(num_var, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_ineq, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["b"], ["x"], name="smap")
    # define rounding model
    rnd = roundSTEModel(param_keys=["b"], var_keys=["x"],  output_keys=["x_rnd"], int_ind=model.int_ind, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmQuadratic(["b", "x_rnd"], num_var, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    df = eval(components, model, loader_test)
    if penalty_growth:
        df.to_csv(f"result/cq_ste{penalty_weight}_{num_var}-{num_ineq}-g.csv")
    else:
        df.to_csv(f"result/cq_ste{penalty_weight}_{num_var}-{num_ineq}.csv")


def eval(components, model, loader_test):
    params, sols, objvals, conviols, elapseds = [], [], [], [], []
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
        x = datapoints["x_rnd"]
        for i in range(len(model.vars["x"])):
            model.vars["x"][i].value = x[0,i].item()
        # get solutions
        xval, objval = model.get_val()
        params.append(list(b.cpu().numpy()))
        sols.append(list(list(xval.values())[0].values()))
        objvals.append(objval)
        conviols.append(sum(model.cal_violation()))
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param":params,
                       "Sol":sols,
                       "Obj Val": objvals,
                       "Constraints Viol": conviols,
                       "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solution: {}".format(np.sum(df["Constraints Viol"] > 0)))
    return df
