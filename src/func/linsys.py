"""
Linear system solver to complete solution
"""
import torch
from torch import nn

class completePartial(nn.Module):
    """
    A module to complete partial solution for A x = b system
    """
    def __init__(self, A, num_var, partial_ind, var_key, rhs_key, output_key, name="Complete"):
        super(completePartial, self).__init__()
        # size
        self.num_var = num_var
        # index
        self.partial_ind = partial_ind
        self.other_ind = [i for i in range(num_var) if i not in partial_ind]
        # necessary computation
        self.A = torch.from_numpy(A).float()
        self._A_partial = self.A[:, :num_var//2]
        self._A_other_inv = torch.inverse(self.A[:, num_var//2:])
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
        # complete vars
        x_comp = torch.zeros(x.shape[0], self.num_var, device=x.device)
        x_comp[:, :self.num_var//2] = x
        x_comp[:, self.num_var//2:] = (b - x @ self._A_partial.T) @ self._A_other_inv.T
        data[self.out_key] = x_comp
        return data
