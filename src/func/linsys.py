"""
Linear system solver to complete solution
"""
import torch
from torch import nn
from torch.linalg import lu_factor, lu_solve

class completePartial(nn.Module):
    """
    A module to complete partial solution for A x = b system
    """
    def __init__(self, A, num_var, partial_ind, var_key, rhs_key, output_key, name="Complete"):
        super(completePartial, self).__init__()
        # size
        self.num_var = num_var
        # index
        self.A = A
        self.partial_ind = partial_ind
        self.other_ind = [i for i in range(num_var) if i not in partial_ind]
        # precompute LU decomposition
        self.lu, self.pivots = lu_factor(A[:, self.other_ind].double())
        # keys
        self.x_key = var_key
        self.b_key = rhs_key
        self.out_key = output_key

    def forward(self, data):
        """
        Forward pass to complete partial solution
        """
        # get values
        x, b = data[self.x_key], data[self.b_key]
        rhs = (b - (x @ self.A[:, self.partial_ind].T)).double()
        # complete vars
        x_comp = torch.zeros(x.shape[0], self.num_var, device=x.device)
        x_comp[:, self.partial_ind] = x
        x_comp[:, self.other_ind] = lu_solve(self.lu, self.pivots, rhs.T).T.float()
        data[self.out_key] = x_comp
        return data
