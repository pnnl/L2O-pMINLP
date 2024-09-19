import torch
from torch import nn
from torch.autograd import Function

class thresholdBinarize(nn.Module):
    """
    smoothly rounds the elements in `x` based on the corresponding values in `threshold`
    using a sigmoid function.
    """
    def __init__(self, slope=10):
        super(thresholdBinarize, self).__init__()
        self.slope = slope

    def forward(self, x, threshold):
        # ensure the threshold_tensor values are between 0 and 1
        threshold = torch.clamp(threshold, 0, 1)
        # hard rounding
        hard_round = (x >= threshold).float()
        # calculate the difference and apply the sigmoid function
        diff = self.slope * (x - threshold)
        smoothed_round = torch.sigmoid(diff)
        # return with STE grad
        return hard_round + (smoothed_round - smoothed_round.detach())


class diffGumbelBinarize(nn.Module):
    """
    An autograd model to binarize numbers under Gumbel-Softmax trick
    """
    def __init__(self, temperature=1.0):
        super(diffGumbelBinarize, self).__init__()
        self.temperature = temperature

    def forward(self, x):
        # train mode
        if self.training:
            x = _gumbelSigmoidFuncSTE.apply(x, self.temperature)
        # eval mode
        else:
            x = (torch.sigmoid(x) > 0).float()
        return x


class _gumbelSigmoidFuncSTE(Function):
    @staticmethod
    def forward(ctx, logit, temperature=1.0):
        # Gumbel-Sigmoid sampling
        eps = 1e-20
        gumbel_noise_0 = -torch.log(-torch.log(torch.rand_like(logit) + eps) + eps)
        gumbel_noise_1 = -torch.log(-torch.log(torch.rand_like(logit) + eps) + eps)
        # sigmoid with Gumbel
        noisy_diff = logit + gumbel_noise_1 - gumbel_noise_0
        soft_sample = torch.sigmoid(noisy_diff / temperature)
        # rounding
        hard_sample = torch.round(soft_sample)
        # store soft sample for backward pass
        ctx.save_for_backward(soft_sample)
        return hard_sample

    @staticmethod
    def backward(ctx, grad_output):
        soft_sample, = ctx.saved_tensors
        grad_input = grad_output * soft_sample * (1 - soft_sample)
        return grad_input, None


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

    # add system path
    import sys
    import os
    sys.path.append(os.path.abspath("."))
    sys.path.append(os.path.abspath(".."))

    x = torch.tensor([0.2, 0.5, 0.8])
    v = torch.tensor([0.3, 0.5, 0.7], requires_grad=True)
    round_func = thresholdBinarize()
    print("Input tensor:", x)
    print("Threshold:", v)
    res = round_func(x, v)
    print(res)
    res.backward(torch.ones_like(res))
    print("Grad:", v.grad)
    print()

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

    # binarize
    bin = diffGumbelBinarize()
    x_bin = bin(x)
    print("Gumbel Binarize vector", x_bin)
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
