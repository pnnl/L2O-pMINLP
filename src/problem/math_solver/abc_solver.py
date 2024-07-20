"""
Parametric Mixed Integer Nonlinear Programming with SCIP
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
import copy

from pyomo import environ as pe
from pyomo import opt as po
from pyomo.core import TransformationFactory

class abcParamSolver(ABC):
    @abstractmethod
    def __init__(self, timelimit=None):
        # create a scip solver
        self.opt = po.SolverFactory("scip")
        # set timelimit
        if timelimit:
            self.opt.options["limits/time"] = timelimit
        # init attributes
        self.model = None # pyomo model
        self.params ={} # dict for pyomo mutable parameters
        self.vars = {} # dict for pyomo decision variable
        self.cons = None # pyomo constraints

    @property
    def int_ind(self):
        """
        Identify indices of integer variables
        """
        int_ind = {}
        for key, vars in self.vars.items():
            int_ind[key] = []
            for i, v in vars.items():
                if v.domain is pe.Integers:
                    int_ind[key].append(i)
        return int_ind

    @property
    def bin_ind(self):
        """
        Identify indices of binary variables
        """
        bin_ind = {}
        for key, vars in self.vars.items():
            bin_ind[key] = []
            for i, v in vars.items():
                if v.domain is pe.Binary:
                    bin_ind[key].append(i)
        return bin_ind


    def solve(self, max_iter=None, tee=False, keepfiles=False):
        """
        Solve the model and return variable values and the objective value
        """
        # solve the model
        self.res = self.opt.solve(self.model, tee=tee, keepfiles=keepfiles)
        # get variable values and objective value
        xval, objval = self.get_val()
        return xval, objval

    def set_param_val(self, param_dict):
        """
        Set values for mutable parameters in the model
        """
        # iterate through different parameters categories
        for key, val in param_dict.items():
            param = self.params[key]
            # iterate each parameter to update value
            if isinstance(val, Iterable):
                for i, v in zip(param, val):
                    param[i].set_value(v)
            # set single value
            else:
                param.set_value(val)

    def get_val(self):
        """
        Retrieve the values of decision variables and the objective value
        """
        # init dict for var value
        solvals = {}
        # get variable values as dict
        try:
            for key, vars in self.vars.items():
                solvals[key] = {i:vars[i].value for i in vars}
            # get the objective value
            objval = pe.value(self.model.obj)
        except:
            # no value
            solvals, objval = None, None
        return solvals, objval

    def check_violation(self):
        """
        Check for any constraint violations in the model
        """
        return any(self._constraint_violation(constr) != 0 for constr in self.model.cons.values())

    def cal_violation(self):
        """
        Calculate the magnitude of violations for each constraint
        """
        return [self._constraint_violation(constr) for constr in self.model.cons.values()]

    def _constraint_violation(self, constr):
        """
        Helper method to compute the violation of a single constraint
        """
        lhs = pe.value(constr.body)
        # check if LHS is below the lower bound
        if constr.lower is not None and lhs < pe.value(constr.lower) - 1e-5:
            return float(pe.value(constr.lower)) - lhs
        # check if LHS is above the upper bound
        elif constr.upper is not None and lhs > pe.value(constr.upper) + 1e-5:
            return lhs - float(pe.value(constr.upper))
        return 0.0

    def clone(self):
        """
        Creates and returns a deep copy of the model
        """
        # shallow copy
        model_new = copy.deepcopy(self)
        # clone pyomo model
        model_new.model = model_new.model.clone()
        # clone variables
        model_new.vars = {var: getattr(model_new.model, var) for var in self.vars}
        # clone constraints
        model_new.cons = model_new.model.cons
        # clone parameters
        model_new.params = {param: getattr(model_new.model, param) for param in self.params}
        return model_new

    def relax(self):
        """
        Relaxe binary & integer variables to continuous variables and returns the relaxed model
        """
        # clone pyomo model
        model_rel = self.clone()
        # iterate through decision variables
        TransformationFactory("core.relax_integer_vars").apply_to(model_rel.model)
        # change solver to ipopt
        #model_rel.opt = po.SolverFactory("ipopt")
        # set number of iterations
        model_rel.opt.options["limits/solutions"] = 100
        # remove timelimit
        model_rel.opt.options["limits/time"] = 1e+20
        return model_rel

    def penalty(self, weight):
        """
        Create a penalty model from an original model to handle constraints as soft constraints
        """
        # clone pyomo model
        model_pen = self.clone()
        model = model_pen.model
        # slacks
        model.slack = pe.Var(pe.Set(initialize=model.cons.keys()), domain=pe.NonNegativeReals)
        # add slacks to objective function as penalty
        penalty = sum(weight * model.slack[s] for s in model.slack)
        obj = model.obj.expr + penalty
        sense = model.obj.sense
        model.del_component(model.obj) # delete original obj
        model.obj = pe.Objective(sense=sense, expr=obj)
        # constraints
        for c in model.slack:
            # deactivate hard constraints
            model.cons[c].deactivate()
            #  modify constraints to incorporate slacks
            if model.cons[c].equality:
                # + - slack ==
                model.cons.add(model.cons[c].body + model.slack[c] >= model.cons[c].lower)
                model.cons.add(model.cons[c].body - model.slack[c] <= model.cons[c].upper)
            elif model.cons[c].lower is not None:
                # + slack >=
                model.cons.add(model.cons[c].body + model.slack[c] >= model.cons[c].lower)
            else:
                # - slack <=
                model.cons.add(model.cons[c].body - model.slack[c] <= model.cons[c].upper)
        return model_pen

    def first_solution_heuristic(self, nodes_limit=1):
        """
        Create a model that terminates after finding the first feasible solution
        """
        # clone pyomo model
        model_heur = self.clone()
        # set solution limit
        model_heur.opt.options["limits/solutions"] = nodes_limit
        return model_heur
