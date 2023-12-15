"""
Parametric Mixed Integer Quadratic Programming
"""

from abc import ABC, abstractmethod
import copy

from pyomo import environ as pe
from pyomo import opt as po
from pyomo.core import TransformationFactory

class abcParamModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @property
    def intInd(self):
        int_ind = []
        for i in self.x:
            if self.x[i].domain is pe.Integers:
                int_ind.append(i)
        return int_ind

    def solve(self, solver, max_iter=None, tee=False):
        """
        Solve the model using the given solver.
        """
        # create a solver instance
        opt = po.SolverFactory(solver)
        if max_iter:
            if solver == "ipopt":
                opt.options["max_iter"] = max_iter
            if solver == "glpk":
                opt.options["itlim"] = max_iter
            if solver == "scip":
                opt.options["limits/totalnodes"] = max_iter
            if solver == "gurobi":
                opt.options["NodeLimit"] = max_iter
            if solver == "cplex":
                opt.options["mip.limits.nodes"] = max_iter
            if solver == "cbc":
                opt.options["maxNodes"] = max_iter
        # solve the model
        self.res = opt.solve(self.model, tee=tee)#, keepfiles=True)
        # variable values and objective value
        xval, objval = self.getVal()
        return xval, objval

    def setParamValue(self, *values):
        ind = 0
        for param in self.params.values():
            for p in param:
                param[p].set_value(values[ind])
                ind += 1

    def getVal(self):
        # get variable values
        xval = {i:self.x[i].value for i in self.x}
        # calculate the objective value
        objval = pe.value(self.model.obj)
        return xval, objval

    def checkViolation(self):
        """
        Check if any violations in constraints
        """
        return any(self._constraint_violation(constr) != 0 for constr in self.model.cons.values())

    def calViolation(self):
        """
        Calculate the violations for each constraint.
        """
        return [self._constraint_violation(constr) for constr in self.model.cons.values()]

    def _constraint_violation(self, constr):
        """
        Helper method to compute the violation of a single constraint.
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
        Copy the model
        """
        # shallow copy
        model_new = copy.copy(self)
        # clone pyomo model
        model_new.model = model_new.model.clone()
        # clone variables
        model_new.x = {}
        ind = 0
        for v in model_new.model.component_objects(pe.Var, active=True):
            for i in v:
                model_new.x[ind] = v[i]
                ind += 1
        # clone constraints
        model_new.cons = model_new.model.cons
        model_new.params = {param: getattr(model_new.model, param) for param in self.params}
        return model_new

    def relax(self):
        """
        Change the domain of the integer variables to continuous
        """
        # clone pyomo model
        model_rel = self.clone()
        # relax
        TransformationFactory('core.relax_integer_vars').apply_to(model_rel.model)
        return model_rel

    def penalty(self, weight):
        """
        Create a penalty model from an original model
        """
        # clone pyomo model
        model_pen = self.clone()
        model = model_pen.model
        # slacks
        model.slack = pe.Var(pe.Set(initialize=model.cons.keys()), domain=pe.NonNegativeReals)
        # add slacks to objective function as penalty
        obj = model.obj.expr + sum(weight * model.slack[s] for s in model.slack)
        sense = model.obj.sense
        model.del_component(model.obj)
        model.obj = pe.Objective(sense=sense, expr=obj)
        # constraints
        for c in model.slack:
            # deactivate constraint
            model.cons[c].deactivate()
            # penalty
            if model.cons[c].equality: # ==
                model.cons.add(model.cons[c].body + model.slack[c] >= model.cons[c].lower)
                model.cons.add(model.cons[c].body - model.slack[c] <= model.cons[c].upper)
            elif model.cons[c].lower is not None: # >=
                model.cons.add(model.cons[c].body + model.slack[c] >= model.cons[c].lower)
            else: # <=
                model.cons.add(model.cons[c].body - model.slack[c] <= model.cons[c].upper)
        return model_pen