# Differentiable Mixed-Integer Programming Layers

This project is an implementation of Differentiable Mixed-Integer Programming Layers based on the [NeuroMANCER library](https://github.com/pnnl/neuromancer). It introduces learnable differentiable correction layers for rounding and projection, enabling efficient integer solution acquisition for parametric nonlinear mixed-integer problems through neural networks, thus eliminating the need for traditional mathematical programming solvers.

While inherently heuristic and not guaranteed to find the optimal or even a feasible solution, the framework often provides high-quality feasible solutions that are extremely useful either as alternatives to optimal solutions or as initial solutions for traditional solvers. This capability makes them invaluable tools in complex optimization scenarios where exact methods might struggle or be too slow.

## Features

- **Efficient Solution Acquisition**: The entire solution process relies entirely on neural networks without the need for mathematical programming solvers.

- **Integer Solution Guarantee**: Integrates learnable rounding directly into the network architecture, ensuring that solutions adhere strictly to integer constraints.

## Problem Definition

A generic formulation of a multiparametric mix-integer nonlinear program (pMINLP) is given in the form:

$$
\begin{aligned}
  \underset{\boldsymbol{\Theta}}{\min} \quad & \frac{1}{m} \sum_{i=1}^m  f(\mathbf{x}^i, \mathbf{y}^i, \boldsymbol{\xi}^i) \\
  s.t. \quad & \mathbf{g} (\mathbf{x}^i, \mathbf{y}^i, \boldsymbol{\xi}^i) \leq \mathbf{0} \quad \forall i \\
  & \mathbf{x}^i \in \mathbb{R}^{n_R} \quad \forall i \\
  & \mathbf{y}^i \in \mathbb{Z}^{n_I} \quad \forall i \\
  & [\mathbf{x}^i, \mathbf{y}^i] = \boldsymbol{\pi}_{\boldsymbol{\Theta}} （\boldsymbol{\xi}^i）\quad \forall i \\
  & \boldsymbol{\xi}^i \in \boldsymbol{\Xi} \subset \mathbb{R}^s  \forall i
\end{aligned}
$$

where $\boldsymbol{\Xi}$ represents the sampled dataset and $\boldsymbol{\xi}^i$ denotes the $i$-th sample. The vector $\mathbf{x}^i \in \mathbb{R}^{n_R}$ represents the continuous variables, and $\mathbf{y}^i \in \mathbb{Z}^{n_I}$ represents the integer variables, both of which are involved in minimizing the objective function $f(\cdot)$ while satisfying a set of inequality constraints $\mathbf{g}(\cdot) \leq 0$. The mapping $\boldsymbol{\pi}_{\boldsymbol{\Theta}}(\boldsymbol{\xi}^i)$, given by a deep neural network parametrized by $\Theta$, represents the solution to the optimization problem.


