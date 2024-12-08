
"""
Definition of neuromancer.SoftBinary class and neuromancer.SoftInteger
imposing soft constraints on to approximate binary and integer variables
    x in Z
    x in N
"""

from abc import ABC, abstractmethod
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuromancer.component import Component
from neuromancer.gradients import gradient


def sawtooth_round(value):
    # https: // en.wikipedia.org / wiki / Sawtooth_wave
    x = (torch.atan(torch.tan(np.pi * value))) / np.pi
    return x


def sawtooth_floor(value):
    x = (torch.atan(torch.tan(np.pi * value + 0.5*np.pi))) / np.pi + 0.5
    return x


def sawtooth_ceil(value):
    x = (-np.pi+torch.atan(torch.tan(np.pi * value + 0.5*np.pi))) / np.pi + 0.5
    return x


def smooth_sawtooth_round(value):
    x = (torch.tanh(torch.tan(np.pi * value))) / np.pi
    return x


def smooth_sawtooth_floor(value):
    x = (torch.tanh(torch.tan(np.pi * value + 0.5*np.pi))) / np.pi + 0.5
    return x


def smooth_sawtooth_ceil(value):
    x = (-np.pi+torch.tanh(torch.tan(np.pi * value + 0.5*np.pi))) / np.pi + 0.5
    return x


def smooth_sine_integer(value):
    # https://connectionism.tistory.com/100
    x = torch.sin(2*np.pi*value)/(2*np.pi)
    return x


class VarConstraint(Component, ABC):
    def __init__(self, input_keys, output_keys=[], name=None):
        """
        VarConstraint is canonical Component class for imposing binary and integer constraints on variables in the input_keys
        :param input_keys: (dict {str: str}) Input keys of variables to be constrained, e.g., ["x", "y", "z"]
        :param output_keys: [str] optional list of strings to define new variable keys at the output,
                            by default input_keys are being used, thus the original input values are being overwritten
        """
        input_keys = input_keys if isinstance(input_keys, list) else [input_keys]
        if bool(output_keys):
            output_keys = output_keys if isinstance(output_keys, list) else [output_keys]
            assert len(output_keys) == len(input_keys), \
                f'output_keys must have the same number of elements as input_keys. \n' \
                f'{len(output_keys)} output_keys were given, but {len(input_keys)} were expected.'
        else:
            output_keys = input_keys
        super().__init__(input_keys=input_keys, output_keys=output_keys, name=name)

    @abstractmethod
    def transform(self, x):
        pass

    def forward(self, data):
        output_dict = {}
        for key_in, key_out in zip(self.input_keys, self.output_keys):
            output_dict[key_out] = self.transform(data[key_in])
        return output_dict


class SoftBinary(VarConstraint):
    def __init__(self, input_keys, output_keys=[], threshold=0.0, scale=10., name=None):
        """
        SoftBinary is class for imposing soft binary constraints on input variables in the input_keys list
        :param input_keys: (dict {str: str}) Input keys of variables to be constrained, e.g., ["x", "y", "z"]
        :param output_keys: [str] optional list of strings to define new variable keys at the output,
                            by default input_keys are being used, thus the original input values are being overwritten
        :param scale: float, scaling value for better conditioning of the soft binary approximation
        :param name:
        """
        super().__init__(input_keys=input_keys, output_keys=output_keys, name=name)
        self.scale = scale
        self.threshold = threshold

    def transform(self, x):
        return torch.sigmoid(self.scale*(x - self.threshold))


class IntegerProjection(VarConstraint):
    step_methods = {'round_sawtooth': sawtooth_round,
                    'round_smooth_sawtooth': smooth_sawtooth_round,
                    'ceil_sawtooth': sawtooth_ceil,
                    'ceil_smooth_sawtooth': smooth_sawtooth_ceil,
                    'floor_sawtooth': sawtooth_floor,
                    'floor_smooth_sawtooth': smooth_sawtooth_floor,
                    }

    def __init__(self, input_keys, output_keys=[], method="round_sawtooth", nsteps=1, stepsize=0.5, name=None):
        """

        IntegerCorrector is class for imposing differentiable integer correction on input variables in the input_keys list
        :param input_keys: (dict {str: str}) Input keys of variables to be constrained, e.g., ["x", "y", "z"]
        :param output_keys: [str] optional list of strings to define new variable keys at the output,
                            by default input_keys are being used, thus the original input values are being overwritten
        :param method: [str] integer correction step method ['sawtooth', 'smooth_sawtooth']
        :param nsteps:
        :param name:
        """
        super().__init__(input_keys=input_keys, output_keys=output_keys, name=name)
        self.step = self._set_step_method(method)
        self.nsteps = nsteps
        self.stepsize = stepsize

    def _set_step_method(self, method):
        if method in self.step_methods:
            return self.step_methods[method]
        else:
            assert callable(method), \
                f'The integer correction step method, {method} must be a key in {self.step_methods} ' \
                f'or a differentiable callable.'
            return method

    def transform(self, x):
        for k in range(self.nsteps):
            x = x - self.stepsize*self.step(x)
        return x


class BinaryProjection(IntegerProjection):
    def __init__(self, input_keys, output_keys=[], threshold=0.0, scale=1.,
                 method="round_sawtooth", nsteps=1, stepsize=0.5, name=None):
        """
        BinaryProjection is class for imposing binary constraints correction on input variables in the input_keys list
        generates: x in {0, 1}
        if x <  threshold then x = 0
        if x >= threshold then x =1
        :param input_keys: (dict {str: str}) Input keys of variables to be constrained, e.g., ["x", "y", "z"]
        :param output_keys: [str] optional list of strings to define new variable keys at the output,
                            by default input_keys are being used, thus the original input values are being overwritten
        :param scale: float, scaling value for better conditioning of the soft binary approximation
        :param name:
        """
        super().__init__(input_keys=input_keys, output_keys=output_keys, method=method,
                         nsteps=nsteps, stepsize=stepsize, name=name)
        self.scale = scale
        self.threshold = threshold

    def transform(self, x):
        x = torch.sigmoid(self.scale * (x - self.threshold))
        for k in range(self.nsteps):
            x = x - self.stepsize * self.step(x)
        return x


class IntegerInequalityProjection(IntegerProjection):
    def __init__(self, constraints, input_keys,
                 method="sawtooth", direction='gradient', dropout=0.,
                 n_projections=1, viol_tolerance=1e-3, proj_grad_stepsize=0.01,
                 nsteps=1, stepsize=0.5, batch_second=False, name=None):
        """
        Implementation of projected gradient method for corrections of integer constraints violations
        Algorithm
            input: decision variable x
            do: projection to the nearest integer
            for step in range(n_projections):
                do: evaluate constraints violation energy (magnitude of viols per sample)
                do: if no violations then terminate
                do: else calculate directions of constraints corrections via constraints gradients or random search
                do: random dropout of the directions
                do: MIP projection
                    for k in range(nsteps):
                        do: projected gradient: x = x - stepsize*direction
                        do: compute integer directions: int_direction via ceil or floor based on sign(direction)
                        do: integer gradient: x = x - stepsize*int_direction
            return x

        :param constraints: list of objects which implement the Loss interface (e.g. Objective, Loss, or Constraint)
        :param input_keys: (List of str) List of input variable names
        :param method: (str) selecting method from list of available integer projection methods
        :param direction: (str) selecting constraints projection direction method
        :param dropout: (float) ratio of random directions to be dropped
        :param n_projections: (int) number of outer loop to calculate integer projections directions
        :param nsteps: (int) number of iteration steps for the inner loop of projections
        :param stepsize: (float) scaling factor for projection updates
        :param batch_second:
        :param name:
        """
        super().__init__(input_keys=input_keys, nsteps=nsteps, stepsize=stepsize, name=name)
        self.constraints = nn.ModuleList(constraints)
        self.batch_second = batch_second
        self._constraints_check()
        self.get_direction = {'gradient': self.get_direction_gradient,
                              'random': self.get_direction_random}[direction]
        self.round_step = self._set_step_method('round_' + method)
        self.ceil_step = self._set_step_method('ceil_' + method)
        self.floor_step = self._set_step_method('floor_' + method)
        self.dropout = dropout
        self.proj_grad_stepsize = proj_grad_stepsize
        self.viol_tolerance = viol_tolerance
        self.n_projections = n_projections
        self.output_keys = ['n_projections']
        for key in self.input_keys:
            for k in range(self.n_projections+2):
                self.output_keys.append(key + f'_{k}')

    def _constraints_check(self):
        for con in self.constraints:
            assert str(con.comparator) in ['lt', 'gt'], \
                f'constraint {con} must be inequality (lt or gt), but it is {str(con.comparator)}'

    def int_projection(self, x):
        """
        projection to the nearest integer
        :param x: tensor
        :return: tensor
        """
        for k in range(self.nsteps):
            x = x - self.stepsize*self.round_step(x)
        return x

    def int_ineq_projection(self, x, mask, direction):
        """
        projection to integer in the direction of the constraints projection
        :param x: (tensor) primal integer decision variable
        :param mask: mask of infeasible samples
        :param direction: constraints gradient w.r.t. x
        :return: (tensor) projected primal integer decision variable
        """
        floor_mask = direction > 0
        ceil_mask = direction < 0
        for k in range(self.nsteps):
            x = x - mask.view(-1, 1)*self.proj_grad_stepsize*direction
            ceil_step = ceil_mask * self.ceil_step(x)
            floor_step = floor_mask * self.floor_step(x)
            step = ceil_step + floor_step
            x = x - mask.view(-1, 1)*self.stepsize*step
        return x

    def con_viol_energy(self, input_dict):
        """
        Calculate the constraints violation potential energy over batches
        """
        C_violations = []
        for con in self.constraints:
            output = con(input_dict)
            cviolation = output[con.output_keys[2]]
            if self.batch_second:
                cviolation = cviolation.transpose(0, 1)
            cviolation = cviolation.reshape(cviolation.shape[0], -1)
            C_violations.append(cviolation)
        C_violations = torch.cat(C_violations, dim=-1)
        energy = torch.mean(torch.abs(C_violations), dim=1)
        return energy

    def get_direction_gradient(self, energy, x):
        step = gradient(energy, x)
        direction = step/torch.abs(step)
        direction[direction != direction] = 0  # replacing nan with 0
        return direction

    def get_direction_random(self, energy, x):
        direction = torch.randn(x.shape)
        return direction

    def forward(self, input_dict):
        output_dict = {}
        # Step 1: get to nearest integer via sawtooth integer projection
        for key in self.input_keys:
            output_dict[key+'_0'] = input_dict[key]
            input_dict[key] = self.int_projection(input_dict[key])
            # TODO: interior point projection
            #   assumption: convergence of the fractional point to feasible region
            #   do standard int_projection
            #   check if after this int projection we violate constraints
            #   if so take projection towards the constraints interior
            #   from the original fractional point based on the directions of
            #   combined constraints and loss gradient
            #   currently we are doing projection of exterior points to find feasible points
            output_dict[key+'_1'] = input_dict[key]
        output_dict['n_projections'] = 0
        for k in range(self.n_projections):
            # Step 2: check for con viol for variables if all zero terminate
            energy = self.con_viol_energy(input_dict)
            mask = energy > self.viol_tolerance
            output_dict['n_projections'] = k
            if energy.sum() == 0.0:
                for key in self.input_keys:
                    for j in range(k, self.n_projections):
                        output_dict[key + f'_{j + 2}'] = input_dict[key]
                break
            # Step 3: calculate directions and project onto feasible region
            for key in self.input_keys:
                # Step 3a, get the gradient constraints violation directions
                direction = self.get_direction(energy, input_dict[key])
                # TODO: averaged direction of constraints and objectives
                if self.dropout:
                    # Step 3b, random dropout of the directions
                    dropout = torch.bernoulli(self.dropout*torch.ones(direction.shape)).to(direction.device)
                    direction = dropout*direction
                # Step 3bc, projection
                input_dict[key] = self.int_ineq_projection(input_dict[key], mask, direction)
                output_dict[key + f'_{k+2}'] = input_dict[key]
        return output_dict


def generate_truth_table(num_binaries):
    table = list(itertools.product([0, 1], repeat=num_binaries))
    return torch.tensor(table)


def generate_MIP_con(num_binaries, tolerance=0.01):
    """
    Function for generating the coefficients of the mixed-integer (MIP) constraints A x_b >= b
    for encoding truth tables for binary variables x_b
    see Table 3.6 in: https://www.uiam.sk/assets/fileAccess.php?id=1305&type=1
    :param num_binaries: number of binary variables x_b
    :param tolerance: numerical tolerance for soft approximation of single binary variable x_b
    :return: A [tensor], b [tensor] as coefficients of the MIP constraints A x_b >= b
    """
    # A is defined by truth table given the number of binary variables num_binaries
    A = torch.tensor(list(itertools.product([0, 1], repeat=num_binaries)))
    A[A == 0] = -1
    # b is the right hand side of the generated MIP inequality A >= b
    b = A.sum(dim=1) - num_binaries*tolerance
    return A, b


def binary_encode_integer(min, max):
    """
    Function to generate binary encoding of an integer in the interval [min, max]
    generates coefficients A, b, of the mixed-integer (MIP) constraints A x_b >= b
    and corresponding vector integer values as tensors
    :param min:
    :param max:
    :return:
    """
    int_min = int(np.ceil(min))
    int_max = int(np.floor(max))
    int_range = int_max - int_min +1
    if int_range > 1:
        num_binaries = int(np.ceil(np.log2(int_range)))
        A, b = generate_MIP_con(num_binaries, tolerance=0.01)
        A = A[:int_range,:].float()
        b = b[:int_range]
        int_values = torch.arange(int_min, int_max+1)
        return A, b, int_values
    else:
        print('There is only one possible integer value to be encoded.')
        print(f'For this purpose use SoftBinary class for switching between 0 '
              f'and desired integer value x in the range {min} <= x <= {max}.')


def soft_binary_to_integer(binary, A, b, int_values, tolerance=0.01):
    """
    Converts the soft binary tensor into integer tensor
        soft_index = min(ReLU(b-A*binary), 1)
        soft_index should have only one nonzero element,
        bit this condition might be violated due to constraints tolerances
        alternative encoding for single hot encoding:
        soft_index = softmax(ReLU(b-A*binary)
        integer = int_values*soft_index
        for inference
        integer = int_values*ceil(soft_index)

    :param binary: Tensor, soft approximation of the binary variable
    :param A: Tensor, matrix of the MIP constraints
    :param b: Tensor, right hand side of the MIP constraints
    :param int_values: Tensor,  admissible values of the integer variables
            given by binary encoding via truth tables
    :param tolerance: numerical tolerance for soft approximation of single binary variable x_b
    :return: Tensor, integer valued tensor
    """
    offset = torch.ceil(F.relu(b-torch.matmul(A, binary)))*binary.shape[0]*tolerance
    soft_index = F.relu(b-torch.matmul(A, binary)) + offset
    int_values_masked = int_values*(soft_index)
    idx = (int_values_masked != 0)
    soft_integer = int_values_masked[idx]
    integer = int_values[idx]
    return soft_integer, integer



"""
objective gradients ideas:
    1, combine constraints gradients with objective gradients - check for agreed signs
    2, after finding feasible point keep going and track the objective and constraints value
    3, constrain local search feasible space: 
        3a, hypercupe arround the first integer
        3b, linear cutting planes: discard (assumption convexity of constraints)
    4, search over different random initializations of the weights of neural networks 
    5, barrier integer projection: round towards nearest integer 
        taking into consideration barrier function values and their gradients
        instead of projecting back to the feasible space, 
        search for admissible integers within the feasible space
        interior point local integer search
        5a, get fractional value of decision variable x
        5b, get hypercube of integers arround x
        5c, drop infeasible points for which constr viol >0 
            option 1: extensive eval of constraints per each point of the hypercube
            option 2: learn small classifier to detect feasible and infeasible points based on a 
            fewer number of constraints evaluations 
    6, instead of path explore always from the x_0 point
        pick one integer at the time from the directions and for each eval constr viol 
        if feasible eval loss and log
        after m samples are evaluated pick the best value
"""

