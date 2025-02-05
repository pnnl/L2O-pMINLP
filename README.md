# Learning-to-Optimize for Mixed-Integer Non-Linear Programming

This repository provides the official implementation of our paper: **"[Learning to Optimize for Mixed-Integer Nonlinear Programming](https://arxiv.org/abs/2410.11061)"**

![Framework](img/pipeline.png)


## Overview

Our approach introduces the first general Learning-to-Optimize (L2O) framework designed for Mixed-Integer Nonlinear Programming (MINLP). As illustrated above, the approach consists of two core components: **integer correction layers** and a **feasibility projection heuristic**. 

Traditional solvers struggle with large-scale MINLPs due to combinatorial complexity and non-convexity. With up to tens of thousands of variables, traditional solvers and heuristics even fail to find any feasible solution. Our framework leverages deep learning to predict high-quality solutions with orders-of-magnitude speedup, enabling optimization in scenarios where exact solvers fail.


## Key Features

- 🤖 **Self-supervised learning**: Eliminates the need for optimal solutions as labels.
- 🔢 **Efficient integer correction**: Ensures integer feasibility through a learnable correction layer.
- 🎯 **Efficient feasibility correction**: Refine constraints violation via a gradient-based post-processing.
- 🚀 **Scalability**: Handles problems with up to **20,000 variables** within subsecond inference.  


## Citation

```
@article{tang2024learning,
  title={Learning to Optimize for Mixed-Integer Non-linear Programming},
  author={Tang, Bo and Khalil, Elias B and Drgo{\v{n}}a, J{\'a}n},
  journal={arXiv preprint arXiv:2410.11061},
  year={2024}
}
```

## Performance Comparison

Our learning-based methods (RC & LT) achieve comparable or even superior performance to exact solvers (EX) while being orders of magnitude faster. The figures below illustrate the impact of penalty weights on feasibility and objective values for smaller-scale problems:

<div align="center">
    <img src="img/cq_s100_penalty.png" alt="Penalty Effect on IQP" width="40%"/>
    <img src="img/rb_s100_penalty.png" alt="Penalty Effect on MIRB" width="40%"/>
</div>

The top plots show the proportion of feasible solutions, while the bottom plots display objective values. For these instances, EX finds the best feasible solutions within **1000 seconds** (leftmost boxplot in the bottom plots), serving as a benchmark. With properly tuned penalty weights, our approach attains **comparable or better objective values within just subsecond**, demonstrating its efficiency and effectiveness.


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
├── test                           # Some testing and visualization with Jupyter notebooks
├── img                            # Image resources for the project
├── src                            # Main source code directory
│   ├── __init__.py                # Initializes the src package
│   ├── heuristic                  # Modules for the heuristics
│       ├── __init__.py            # Initializes the heuristics submodule
│       ├── rounding.py            # Heuristics for rounding
│       └── resolve.py             # Heuristics for resolving some problem
│   ├── func                       # Directory for function modules
│       ├── __init__.py            # Initializes the function submodule
│       ├── layer.py               # Pre-defined neural network layers
│       ├── ste.py                 # Straight-through estimators for non-differentiable operations
│       ├── rnd.py                 # Modules for differentiable and learnable rounding
│       └── proj.py                # Modules for differentiable and learnable projection
│   ├── postprocess                # Modules for the postprocessing
│       ├── __init__.py            # Initializes the postprocessing submodule
│       └── project.py             # postprocessing for the feasibility projection
│   ├── problem                    # Modules for the benchmark of constrained optimization
│       ├── __init__.py            # Initializes the problem submodule
│       ├── math_solver            # Collection of Predefined SCIP solvers
│           ├── __init__.py        # Initializes the mathematical solver submodule
│           ├── abc_solver.py      # Abstract base class for solver implementations
│           ├── quadratic.py       # SCIP model for Integer Quadratic Problems
│           ├── nonconvex.py       # NeuroMANCER map for Integer Non-Convex Problems
│           └── rosenbrock.py      # SCIP model for Mixed-Intger Rosenbrock Problems
│       └── neuromancer            # Collection of Predefined NeuroMANCER maps
│           ├── __init__.py        # Initializes the NeuroMANCER map submodule
│           ├── quadratic.py       # NeuroMANCER map for Integer Quadratic Problems
│           ├── nonconvex.py       # NeuroMANCER map for Integer Non-Convex Problems
│           └── rosenbrock.py      # NeuroMANCER map for Mixed-Intger Rosenbrock Problems
│   └── utlis                      # Utility tools such as data processing and result test
│       ├── __init__.py            # Initializes the utility submodule
│       └── data.py                # Data processing file
│       └── solve_test.py          # Testing functions to evaluate optimization solution
├── run_qp.py                      # Script to run experiments for Integer Quadratic Problems
├── run_nc.py                      # Script to run experiments for Integer Non-Convex Problems
├── run_rb.py                      # Script to run experiments for Mixed-Integer Rosenbrock Problems
└── README.md                      # README file for the project
```


## Reproducibility

Our framework supports three benchmark problems:

- Integer Quadratic Problems (IQP): A convex quadratic objective with linear constraints and integer variables.
- Integer Non-Convex Problems (INP): A more challenging variant incorporating trigonometric terms, introducing non-convexity.
- Mixed-Integer Rosenbrock Problems (MIRB): A highly nonlinear benchmark derived from the Rosenbrock function, with linear and non-linear constraints.

To reproduce experiments, use the following commands:

### Integer Quadratic Problems

```Python
python run_qp.py --size 5
```
### Integer Non-Convex Problems

```Python
python run_nc.py --size 10 --penalty 1 --project
```
### Mixed-Integer Rosenbrock Problems

```Python
python run_rb.py --size 100 --penalty 10 --project
```

### Arguments

- `--size`: Specifies the problem size. Larger values correspond to more decision variables.
- `--penalty`: Sets the penalty weight for constraint violations (default: 100).
- `--project:` Enables feasibility projection as a post-processing step.
