"""
Equality constraint encoding
"""

import numpy as np
from scipy.linalg import null_space
import torch
import torch.nn as nn

class nullSpaceEncoding(nn.Module):
    def __init__(self, A, b, input_key, output_key):
        """
        Neural network module to encode equality constraints A x = b using null space decomposition.
        """
        super().__init__()
        # data keys
        self.input_key = input_key
        self.output_key = output_key
        # sepecial solution for equality constraints
        x_s, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        # null space for equality constraints
        N = null_space(A)
        # to pytorch
        self.x_s = torch.tensor(x_s, dtype=torch.float32).view(1, -1)
        self.N = torch.tensor(N, dtype=torch.float32)
        # init device
        self.device = None

    def forward(self, data):
        # get free parameters
        z = data[self.input_key]
        if z.device != self.device:
            self.device = z.device
            self.x_s = self.x_s.to(self.device)
            self.N = self.N.to(self.device)
        # compute x = x_s + N z^T
        x = self.x_s + torch.einsum("bj,ij->bi", z, self.N)
        data[self.output_key] = x
        return data

if __name__ == "__main__":
    # params
    num_blocks = 10

    # create A&b for A x == b
    rng = np.random.RandomState(17)
    A = rng.normal(scale=1, size=(3, num_blocks))
    b = np.zeros(3)

    # init module
    encoding = nullSpaceEncoding(A, b, input_key="z", output_key="x")

    # example data
    batch_size = 32
    N_dim = 7  # Null space dimension
    z_data = torch.randn(batch_size, N_dim)  # Random free parameters
    data = {"z": z_data}

    # encode the constrained output
    data = encoding(data)
    x = data["x"].detach().numpy()

    # verify the output satisfies A x = b
    for i in range(32):
        print(A @ x[i])
