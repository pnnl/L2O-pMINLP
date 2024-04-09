"""
Parametric Nonlinear Programming with neuroMANCER
"""

from abc import ABC, abstractmethod

import neuromancer as nm

class abcNMProblem(nm.problem.Problem, ABC):
    def __init__(self, vars, params, components, penalty_weight):
        # weight for penalty
        self.penalty_weight = penalty_weight
        # mutable parameters
        self.params = {}
        for p in params:
            self.params[p] = nm.constraint.variable(p)
        # decision variables
        self.vars = {}
        for v in vars:
            self.vars[v] = nm.constraint.variable(v)
        # get obj & constrs
        obj = self.getObj(self.vars, self.params)
        constrs = self.getConstrs(self.vars, self.params, self.penalty_weight)
        # merit loss function
        loss = nm.loss.PenaltyLoss([obj], constrs)
        # optimization solver
        super().__init__(components, loss)

    @abstractmethod
    def getObj(self, vars, params):
        """
        Get neuroMANCER objective component
        """
        pass

    @abstractmethod
    def getConstrs(self, vars, params, penalty_weight):
        """
        Get neuroMANCER constraint component
        """
        pass
