import numpy as np
import torch

x = 10 * torch.rand(5, requires_grad=True)
x.retain_grad()
print("x:", x)
v = torch.rand(5, requires_grad=True) - 0.5
v.retain_grad()
print("v:", v)

print()
x_step= torch.atan(torch.tan(torch.pi * (x + v))) / torch.pi - v
print("diff:", x_step)

print()
for alpha in np.arange(0.1, 1.1, 0.1):
    print("alpha: {:.1f}".format(alpha))
    (x - alpha * x_step).backward(torch.ones_like(x), retain_graph=True)
    print("x_grad:", x.grad)
    print("v_grad:", v.grad)
    # reset gradients
    x.grad.zero_()
    v.grad.zero_()
    print()