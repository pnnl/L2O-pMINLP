import numpy as np
import torch
import matplotlib.pyplot as plt


# select single set of parameters in range [-0.5, 0.5]
x = torch.arange(-2.0, 2.0, 0.01)
value = 0.5
a = value
b = -value
# correction step
x_step = torch.atan(torch.tan(torch.pi*x + a*torch.pi))/torch.pi + b
plt.figure()
plt.plot(x, np.zeros(x_step.shape), 'k--')
plt.plot(x, x_step.detach().numpy())
# rounding
alpha = 1.0
x_round = x - alpha*x_step
plt.figure()
plt.plot(x, np.zeros(x_step.shape), 'k--')
plt.plot(x, x_round.detach().numpy())


"""
search for admissible parameter values
"""
# sample a, b parameters in range [-0.5, 0.5]
a_values = torch.arange(-0.5, 0.6, 0.1)
b_values = torch.arange(-0.5, 0.6, 0.1)

# get the round values for all parameter combinations
X_steps = []
X_rounds = []
ab_tuples = []
for a in a_values:
    for b in b_values:
        x_step = torch.atan(torch.tan(torch.pi * x + a * torch.pi)) / torch.pi + b
        x_round = x - alpha*x_step
        X_steps.append(x_step)
        X_rounds.append(x_round)
        ab_tuples.append((a, b))
steps = torch.stack(X_steps).T
rounds = torch.stack(X_rounds).T

# round scenarios crossing 0.0 value
filter_round = torch.sum(rounds == 0.0, dim=0) > 0
display = sum(filter_round)

# plot desired rounding patters
plt.figure()
plt.plot(x, np.zeros(x_step.shape), 'k--')
plt.plot(torch.tile(x, (display, 1)).T.detach().numpy(),
         steps[:,filter_round].detach().numpy())
plt.figure()
plt.plot(x, np.zeros(x_round.shape), 'k--')
plt.plot(torch.tile(x, (display, 1)).T.detach().numpy(),
         rounds[:,filter_round].detach().numpy())

# get indices of the parameter scenarios
param_idx = filter_round.nonzero().squeeze().tolist()

correct_params = [ab_tuples[i] for i in param_idx]

"""
Parametric round formula:
    x_step = atan(tan(pi * x + value * pi)) / pi - value
    x_round = x - alpha * x_step

Observation:
   continuous change of the value parameter in range [-0.5, 0.5]
  leads to a change in the rounding threshold.

Special cases:
   value = 0.5  -> ceil
   value = 0.0  -> round
   value = -0.5 -> floor
"""

# plot parametric round function with continuous change in value parameter
plt.figure()
plt.plot(x, np.zeros(x.shape), 'k--')
values = torch.arange(-0.5, 0.6, 0.1)
alpha = 1.0  # step size
colors = plt.cm.rainbow(np.linspace(0, 1, len(values)))
for i, value in enumerate(values):
    x_step = torch.atan(torch.tan(torch.pi * x + value * torch.pi)) / torch.pi -value
    x_round = x - alpha * x_step
    plt.step(x, x_round.detach().numpy(), color=colors[i],
             linewidth=2.0, label=f'value={value:.1f}')
plt.legend(loc='lower right')
plt.xlabel('x')
plt.ylabel('parametric round')