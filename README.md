# Differentiable Mixed-Integer Programming Layers

![Framework](img/pipeline.png)

This project is an implementation of Differentiable Mixed-Integer Programming Layers based on the [NeuroMANCER library](https://github.com/pnnl/neuromancer). It introduces learnable differentiable correction layers for rounding and projection, enabling efficient integer solution acquisition for parametric nonlinear mixed-integer problems through neural networks, thus eliminating the need for traditional mathematical programming solvers.

While inherently heuristic and not guaranteed to find the optimal or even a feasible solution, the framework often provides high-quality feasible solutions that are extremely useful either as alternatives to optimal solutions or as initial solutions for traditional solvers. This capability makes them invaluable tools in complex optimization scenarios where exact methods might struggle or be too slow.

## Features

- **Efficient Solution Acquisition**: The entire solution process relies entirely on neural networks without the need for mathematical programming solvers.

- **Integer Solution Guarantee**: Integrates learnable rounding directly into the network architecture, ensuring that solutions adhere strictly to integer constraints.

## Problem Definition

A generic formulation of a multiparametric mix-integer nonlinear program (pMINLP) is given in the form:

$$
\begin{aligned}
  \underset{\boldsymbol{\Theta}}{\min} \quad & \frac{1}{m} \sum_{i=1}^m f(\mathbf{x}^i_R, \mathbf{x}^i_Z, \boldsymbol{\xi}^i) \\ 
  \text{s.t.} \quad 
  & \mathbf{g} (\mathbf{x}^i_R, \mathbf{x}^i_Z, \boldsymbol{\xi}^i) \leq \mathbf{0} \quad \forall i \\ 
  & \mathbf{h} (\mathbf{x}^i_R, \mathbf{x}^i_Z, \boldsymbol{\xi}^i) = \mathbf{0} \quad \forall i \\ 
  & \mathbf{x}^i_R \in \mathbb{R}^{n_r} \quad \forall i \\ 
  & \mathbf{x}^i_Z \in \mathbb{Z}^{n_i} \quad \forall i \\ 
  & [\mathbf{x}^i_R, \mathbf{x}^i_Z] = \boldsymbol{\pi}_{\boldsymbol{\Theta}} (\boldsymbol{\xi}^i) \quad \forall i \\ 
  & \boldsymbol{\xi}^i \in \boldsymbol{\Xi} \subset \mathbb{R}^s \quad \forall i 
\end{aligned}
$$

where $\boldsymbol{\Xi}$ represents the sampled dataset and $\boldsymbol{\xi}^i$ denotes the $i$-th sample. The vector $\mathbf{x}^i_R$ represents the continuous variables, and $\mathbf{x}^i_Z$ represents the integer variables, both of which are involved in minimizing the objective function $f(\cdot)$ while satisfying a set of inequality and equality constraints $\mathbf{g}(\cdot) \leq 0$ and $\mathbf{h}(\cdot) = 0$. The mapping $\boldsymbol{\pi}_{\boldsymbol{\Theta}}(\boldsymbol{\xi}^i)$, given by a deep neural network parametrized by $\Theta$, represents the solution to the optimization problem.

## Requirements

To run this project, you will need the following libraries and software installed:

- **Python**: The project is developed using Python. Ensure you have Python 3.9 or later installed.
- **Scikit-Learn**: Useful for performing various machine learning tasks. 
- **PyTorch**: Used for building and training neural network models.
- **NumPy**: Essential for numerical operations.
- **Pandas**: Useful for data manipulation and analysis.
- **Pyomo**: A Python library for optimization modeling.
- **SCIP**: A powerful solver for mathematical programming, which might need a separate installation process.
- **Neuromancer**: This project uses the Neuromancer library for differentiable programming.

## Code Structure

```
├── archive                        # Archive for older files and documents
├── img                            # Image resources for the project
├── src                            # Main source code directory
│   ├── __init__.py                # Initializes the src package
│   ├── func                       # Directory for function modules
│       ├── __init__.py            # Initializes the function submodule
│       ├── layer.py               # Pre-defined neural network layers
│       ├── ste.py                 # Straight-through estimators for non-differentiable operations
│       ├── rnd.py                 # Modules for differentiable and learnable rounding
│       └── proj.py                # Modules for differentiable and learnable projection
│   ├── problem                    # Modules for the benchmark of constrained optimization
│       ├── __init__.py            # Initializes the problem submodule
│       ├── math_solver            # Collection of Predefined SCIP solvers
│           ├── __init__.py        # Initializes the mathematical solver submodule
│           ├── abc_solver.py      # Abstract base class for solver implementations
│           ├── quadratic.py       # SCIP model for MIQP
│           └── rosenbrock.py      # SCIP model for MIRosenbrock
│       └── neuromancer            # Collection of Predefined NeuroMANCER maps
│           ├── __init__.py        # Initializes the NeuroMANCER map submodule
│           ├── quadratic.py       # NeuroMANCER map for MIQP
│           └── rosenbrock.py      # NeuroMANCER map for MIRosenbrock
│   └── utlis                      # Utility tools such as data processing and result test
│       ├── __init__.py            # Initializes the utility submodule
│       └── data.py                # Data processing file
│       └── solve_test.py          # Testing functions to evaluate optimization solution
├── sweep_QP-Round.py              # Script for hyperparameter tuning for MIQP
├── sweep_Rosenbrock-Round.py      # Script for hyperparameter tuning for MIRosenbrock
└── README.md                      # README file for the project
```

## Parametric MINLP Benchmark

### MIQP

A parametric MIQP model with both continuous variables $\mathbf{x}$ and binary variables $\mathbf{y}$ can be structured as follows:

$$
\begin{aligned}
  \underset{\boldsymbol{\mathbf{x}, \mathbf{y}}}{\min} \quad & \mathbf{c}^\top \mathbf{x} + \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{d}^\top \mathbf{y} \\
  \text{s.t.} \quad 
  & \mathbf{A} \mathbf{x} + \mathbf{E} \mathbf{y} \leq \mathbf{b} + \mathbf{F} \mathbf{\theta} \\
  & \mathbf{x} \geq \mathbf{0} \\
  & \mathbf{y} \in \{0, 1\}
\end{aligned}
$$

In this formulation, the objective function is a quadratic function of $\mathbf{x}$ plus linear in both $\mathbf{x}$ and $\mathbf{y}$. The constraints involve linear combinations of these variables, while the right-hand sides are modulated by the parameter $\mathbf{\theta}$.
