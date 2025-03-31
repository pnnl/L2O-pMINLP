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

def rndCls(loader_train, loader_test, loader_val, config, penalty_growth=False):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"RC in NC for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmNonconvex
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
    step_size = config.step
    # init model
    from src.problem import msNonconvex
    model = msNonconvex(num_var, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_ineq*2, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["b", "d"], ["x"], name="smap")
    # define rounding model
    layers_rnd = netFC(input_dim=num_ineq*2+num_var,
                       hidden_dims=[hsize]*hlayers_rnd,
                       output_dim=num_var)
    rnd = roundGumbelModel(layers=layers_rnd, param_keys=["b", "d"], var_keys=["x"],
                           output_keys=["x_rnd"], int_ind=model.int_ind,
                           continuous_update=True, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmNonconvex(["b", "d", "x_rnd"], num_var, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    df = evaluate(components, loss_fn, model, loader_test, step_size)
    # save csv
    df.to_csv(f"result/nc_cls{penalty_weight}_{num_var}-{num_ineq}-ps{step_size}.csv")


def rndThd(loader_train, loader_test, loader_val, config, penalty_growth=False):
    print(config)
    # random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print(f"LT in NC for size {config.size}.")
    import neuromancer as nm
    from src.problem import nmNonconvex
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
    step_size = config.step
    # init model
    from src.problem import msNonconvex
    model = msNonconvex(num_var, num_ineq, timelimit=1000)
    # build neural architecture for the solution map
    func = nm.modules.blocks.MLP(insize=num_ineq*2, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hsize]*hlayers_sol)
    smap = nm.system.Node(func, ["b", "d"], ["x"], name="smap")
    # define rounding model
    layers_rnd = netFC(input_dim=num_ineq*2+num_var,
                       hidden_dims=[hsize]*hlayers_rnd,
                       output_dim=num_var)
    rnd = roundThresholdModel(layers=layers_rnd, param_keys=["b", "d"], var_keys=["x"],
                              output_keys=["x_rnd"], int_ind=model.int_ind,
                              continuous_update=True, name="round")
    # build neuromancer problem for rounding
    components = nn.ModuleList([smap, rnd]).to("cuda")
    loss_fn = nmNonconvex(["b", "d", "x_rnd"], num_var, num_ineq, penalty_weight)
    # train
    utils.train(components, loss_fn, loader_train, loader_val, lr, penalty_growth)
    # eval
    df = evaluate(components, loss_fn, model, loader_test, step_size)
    # save csv
    df.to_csv(f"result/nc_thd{penalty_weight}_{num_var}-{num_ineq}-ps{step_size}.csv")


def evaluate(components, loss_fn, model, loader_test, project_step_size):
    # postprocessing
    from src.postprocess.project import gradientProjection
    # project
    proj = gradientProjection([components[0]], [components[1]], loss_fn, "x", step_size=project_step_size)
    # init res
    objvals, mean_viols, max_viols, num_viols, num_iters, elapseds = [], [], [], [], [], []
    b_test = loader_test.dataset.datadict["b"][:100]
    d_test = loader_test.dataset.datadict["d"][:100]
    for b, d in tqdm(list(zip(b_test, d_test))):
        # data point as tensor
        datapoints = {"b": torch.unsqueeze(b, 0).to("cuda"),
                      "d": torch.unsqueeze(d, 0).to("cuda"),
                      "name": "test"}
        components.eval()
        tick = time.time()
        with torch.no_grad():
            for comp in components:
                datapoints.update(comp(datapoints))
        # projection
        proj(datapoints)
        tock = time.time()
        # assign params
        model.set_param_val({"b":b, "d":d})
        # assign vars
        x = datapoints["x_rnd"]
        for i in range(len(model.vars["x"])):
            model.vars["x"][i].value = x[0,i].item()
        # get solutions
        xval, objval = model.get_val()
        objvals.append(objval)
        viol = model.cal_violation()
        mean_viols.append(np.mean(viol))
        max_viols.append(np.max(viol))
        num_viols.append(np.sum(viol > 1e-6))
        num_iters.append(proj.num_iters)
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Obj Val": objvals,
                       "Mean Violation": mean_viols,
                       "Max Violation": max_viols,
                       "Num Violations": num_viols,
                       "Num Iterations": num_iters,
                       "Elapsed Time": elapseds})
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    return df
