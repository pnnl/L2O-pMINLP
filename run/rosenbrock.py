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
    print(f"EX for size {config.size}.")
    # config
    steepness = config.steepness
    num_blocks = config.size
    # init model
    from src.problem import msRosenbrock
    model = msRosenbrock(steepness, num_blocks, timelimit=1000)
    # init df
    params, sols, objvals, conviols, elapseds = [], [], [], [], []
    # go through test data
    p_test = loader_test.dataset.datadict["p"]
    a_test = loader_test.dataset.datadict["a"]
    for p, a in tqdm(list(zip(p_test, a_test))):
        # set params
        model.set_param_val({"p":p, "a":a})
        # solve
        tick = time.time()
        try:
            xval, objval = model.solve("scip")
            # eval
            params.append(list(p)+list(a))
            sols.append(list(list(xval.values())[0].values()))
            objvals.append(objval)
            conviols.append(sum(model.cal_violation()))
        except:
            # infeasible
            params.append(list(p)+list(a))
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
    df.to_csv(f"result/rb_exact_{num_blocks}.csv")


def relRnd(loader_test, config):
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RR for size {config.size}.")
    from src.heuristic import naive_round
    # config
    steepness = config.steepness
    num_blocks = config.size
    # init model
    from src.problem import msRosenbrock
    model = msRosenbrock(steepness, num_blocks, timelimit=1000)
    # init df
    params, sols, objvals, conviols, elapseds = [], [], [], [], []
    # go through test data
    p_test = loader_test.dataset.datadict["p"]
    a_test = loader_test.dataset.datadict["a"]
    for p, a in tqdm(list(zip(p_test, a_test))):
        # set params
        model.set_param_val({"p":p, "a":a})
        model_rel = model.relax()
        # solve
        tick = time.time()
        try:
            xval_rel, _ = model_rel.solve("scip")
            xval, objval = naive_round(xval_rel, model)
            # eval
            params.append(list(p)+list(a))
            sols.append(list(list(xval.values())[0].values()))
            objvals.append(objval)
            conviols.append(sum(model.cal_violation()))
        except:
            # infeasible
            params.append(list(p)+list(a))
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
    df.to_csv(f"result/rb_rel_{num_blocks}.csv")


def root(loader_test, config):
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"N1 for size {config.size}.")
    # config
    steepness = config.steepness
    num_blocks = config.size
    # init model
    # init model
    from src.problem import msRosenbrock
    model = msRosenbrock(steepness, num_blocks, timelimit=1000)
    model_heur = model.first_solution_heuristic(nodes_limit=1)
    # init df
    params, sols, objvals, conviols, elapseds = [], [], [], [], []
    # go through test data
    p_test = loader_test.dataset.datadict["p"]
    a_test = loader_test.dataset.datadict["a"]
    for p, a in tqdm(list(zip(p_test, a_test))):
        # set params
        model_heur.set_param_val({"p":p, "a":a})
        # solve
        tick = time.time()
        try:
            xval, objval = model_heur.solve("scip")
            # eval
            params.append(list(p)+list(a))
            sols.append(list(list(xval.values())[0].values()))
            objvals.append(objval)
            conviols.append(sum(model_heur.cal_violation()))
        except:
            # infeasible
            params.append(list(p)+list(a))
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
    df.to_csv(f"result/rb_root_{num_blocks}.csv")


def rndCls(loader_train, loader_test, loader_val, config, penalty_growth=False):
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RC for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmRosenbrock
    from src.func.layer import netFC
    from src.func import roundGumbelModel
    # config
    steepness = config.steepness
    num_blocks = config.size
    hlayers_sol = config.hlayers_sol
    hlayers_rnd = config.hlayers_rnd
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    # init model
    from src.problem import msRosenbrock
    model = msRosenbrock(steepness, num_blocks, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_blocks+1, outsize=2*num_blocks, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["p", "a"], ["x"], name="smap")
    # define rounding model
    layers_rnd = netFC(input_dim=3*num_blocks+1, hidden_dims=[hsize]*hlayers_rnd, output_dim=2*num_blocks)
    rnd = roundGumbelModel(layers=layers_rnd, param_keys=["p", "a"], var_keys=["x"], output_keys=["x_rnd"],
                           int_ind=model.int_ind, continuous_update=True, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmRosenbrock(["p", "a", "x_rnd"], steepness, num_blocks, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    df = eval(components, model, loader_test)
    if penalty_growth:
        df.to_csv(f"result/rb_cls{penalty_weight}_{num_blocks}-g.csv")
    else:
        df.to_csv(f"result/rb_cls{penalty_weight}_{num_blocks}.csv")

def rndThd(loader_train, loader_test, loader_val, config, penalty_growth=False):
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"LT for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmRosenbrock
    from src.func.layer import netFC
    from src.func import roundThresholdModel
    steepness = config.steepness
    num_blocks = config.size
    hlayers_sol = config.hlayers_sol
    hlayers_rnd = config.hlayers_rnd
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    # init model
    from src.problem import msRosenbrock
    model = msRosenbrock(steepness, num_blocks, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_blocks+1, outsize=2*num_blocks, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["p", "a"], ["x"], name="smap")
    # define rounding model
    layers_rnd = netFC(input_dim=3*num_blocks+1, hidden_dims=[hsize]*hlayers_rnd, output_dim=2*num_blocks)
    rnd = roundThresholdModel(layers=layers_rnd, param_keys=["p", "a"], var_keys=["x"],  output_keys=["x_rnd"],
                              int_ind=model.int_ind, continuous_update=True, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmRosenbrock(["p", "a", "x_rnd"], steepness, num_blocks, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    df = eval(components, model, loader_test)
    if penalty_growth:
        df.to_csv(f"result/rb_thd{penalty_weight}_{num_blocks}-g.csv")
    else:
        df.to_csv(f"result/rb_thd{penalty_weight}_{num_blocks}.csv")

def lrnRnd(loader_train, loader_test, loader_val, config, penalty_growth=False):
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RL for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmRosenbrock
    from src.func.layer import netFC
    from src.func import roundThresholdModel
    # config
    steepness = config.steepness
    num_blocks = config.size
    hlayers_sol = config.hlayers_sol
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    # init model
    from src.problem import msRosenbrock
    model = msRosenbrock(steepness, num_blocks, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_blocks+1, outsize=2*num_blocks, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["p", "a"], ["x"], name="smap")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap]).to("cuda")
    loss_fn = nmRosenbrock(["p", "a", "x"], steepness, num_blocks, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    from src.heuristic import naive_round
    params, sols, objvals, conviols, elapseds = [], [], [], [], []
    p_test = loader_test.dataset.datadict["p"]
    a_test = loader_test.dataset.datadict["a"]
    for p, a in tqdm(list(zip(p_test, a_test))):
        # data point as tensor
        datapoints = {"p": torch.tensor(np.array([p]), dtype=torch.float32).to("cuda"),
                      "a": torch.tensor(np.array([a]), dtype=torch.float32).to("cuda"),
                      "name": "test"}
        # infer
        components.eval()
        tick = time.time()
        with torch.no_grad():
            for comp in components:
                datapoints.update(comp(datapoints))
        tock = time.time()
        # assign params
        model.set_param_val({"p":p, "a":a})
        # assign vars
        x = datapoints["x"]
        for i in range(num_blocks*2):
            model.vars["x"][i].value = x[0,i].item()
        # get solutions
        xval_rel, _ = model.get_val()
        xval, objval = naive_round(xval_rel, model)
        params.append(list(p)+list(a))
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
        df.to_csv(f"result/rb_lrn{penalty_weight}_{num_blocks}-g.csv")
    else:
        df.to_csv(f"result/rb_lrn{penalty_weight}_{num_blocks}.csv")

def rndSte(loader_train, loader_test, loader_val, config, penalty_growth=False):
    # random seed
    np.random.seed(2)
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    print(f"RS for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmRosenbrock
    from src.func.layer import netFC
    from src.func import roundSTEModel
    # config
    steepness = config.steepness
    num_blocks = config.size
    hlayers_sol = config.hlayers_sol
    hsize = config.hsize
    lr = config.lr
    penalty_weight = config.penalty
    # init model
    from src.problem import msRosenbrock
    model = msRosenbrock(steepness, num_blocks, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_blocks+1, outsize=2*num_blocks, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["p", "a"], ["x"], name="smap")
    # define rounding model
    rnd = roundSTEModel(param_keys=["p", "a"], var_keys=["x"], output_keys=["x_rnd"],
                        int_ind=model.int_ind, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmRosenbrock(["p", "a", "x_rnd"], steepness, num_blocks, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    df = eval(components, model, loader_test)
    if penalty_growth:
        df.to_csv(f"result/rb_ste{penalty_weight}_{num_blocks}-g.csv")
    else:
        df.to_csv(f"result/rb_ste{penalty_weight}_{num_blocks}.csv")


def eval(components, model, loader_test):
    params, sols, objvals, conviols, elapseds = [], [], [], [], []
    p_test = loader_test.dataset.datadict["p"]
    a_test = loader_test.dataset.datadict["a"]
    for p, a in tqdm(list(zip(p_test, a_test))):
        # data point as tensor
        datapoints = {"p": torch.tensor(np.array([p]), dtype=torch.float32).to("cuda"),
                      "a": torch.tensor(np.array([a]), dtype=torch.float32).to("cuda"),
                      "name": "test"}
        # infer
        components.eval()
        tick = time.time()
        with torch.no_grad():
            for comp in components:
                datapoints.update(comp(datapoints))
        tock = time.time()
        # assign params
        model.set_param_val({"p":p, "a":a})
        # assign vars
        x = datapoints["x_rnd"]
        for i in range(len(model.vars["x"])):
            model.vars["x"][i].value = x[0,i].item()
        # get solutions
        xval, objval = model.get_val()
        params.append(list(p)+list(a))
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