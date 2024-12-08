import torch
from torch import nn
from torch.autograd import Function

class diffBinarize(nn.Module):
    """
    An autograd model to binarize numbers
    """
    def __init__(self):
        super(diffBinarize, self).__init__()

    def forward(self, x):
        x = _binarizeFuncSTE.apply(x)
        return x


class _binarizeFuncSTE(Function):
    """
    An autograd binarize function with straight-through estimator
    """
    @staticmethod
    def forward(ctx, input):
        # clipped ReLU
        logit = torch.clamp(input, min=-1, max=1)
        # save for backward pass
        ctx.save_for_backward(logit)
        # binarize to 0 or 1
        return (logit >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        logit, = ctx.saved_tensors
        grad_input = grad_output * (torch.abs(logit) < 1).float()
        return grad_input


class diffFloor(nn.Module):
    """
    An autograd model to floor numbers
    """
    def __init__(self):
        super(diffFloor, self).__init__()

    def forward(self, x):
        x = _floorFuncSTE.apply(x)
        return x


class _floorFuncSTE(Function):
    """
    An autograd floor function with straight-through estimator
    """
    @staticmethod
    def forward(ctx, input):
        return torch.floor(input).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class solverWrapper(nn.Module):
    """
    A fake autograd model to wrap optimization solver
    """
    def __init__(self, model, solver="ipopt", max_iter=100):
        super(solverWrapper, self).__init__()
        self.model = model
        self.solver = solver
        self.max_iter = max_iter

    def forward(self, params):
        x = _solverFuncFakeGrad.apply(self.model, self.solver, self.max_iter, params)
        return x


class _solverFuncFakeGrad(Function):
    """
    A fake autograd solver with 0 gradient
    """
    @staticmethod
    def forward(ctx, model, solver, max_iter, params):
        # get device
        ctx.device = params.device
        # get shape
        ctx.shape = params.shape
        # convert tensor
        params = params.detach().to("cpu").numpy()
        # get solution
        xvals = []
        for param in params:
            # set parameters
            model.setParamValue(*param)
            # solve
            xval, _ = model.solve(solver=solver, max_iter=max_iter)
            xvals.append(list(xval.values()))
        xvals = torch.FloatTensor(xvals).to(ctx.device)
        return xvals

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.zeros(ctx.shape).to(ctx.device)
        return None, None, None, grad_input


if __name__ == "__main__":

    # input tensor
    x = torch.tensor([[-1.2, -0.5, 0.0, 0.5, 1.2]], requires_grad=True)
    print("Input tensor:", x)

    # floor
    floor = diffFloor()
    x_flr = floor(x)
    print("Floor vector", x_flr)
    loss = x_flr.sum()
    loss.backward()
    print("Grad", x.grad)

    # Reset gradients
    x.grad.zero_()
    print()

    # binarize
    bin = diffBinarize()
    x_bin = bin(x)
    print("Binarize vector", x_bin)
    loss = x_bin.sum()
    loss.backward()
    print("Grad:", x.grad)

    # Reset gradients
    x.grad.zero_()
    print()

    # solver
    from problem.solver import exactQuadratic
    model = exactQuadratic(n_vars=5, n_integers=0)
    solver_func = solverWrapper(model)
    sol = solver_func(x)
    print("Solution:", sol)
    loss = sol.sum()
    loss.backward()
    print("Grad:", x.grad)