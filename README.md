# Differentiable Mixed-Integer Programming Layers

This project is an implementation of Differentiable Mixed-Integer Programming Layers based on the [NeuroMANCER library](https://github.com/pnnl/neuromancer). It introduces learnable differentiable correction layers for rounding and projection, enabling efficient integer solution acquisition for parametric nonlinear mixed-integer problems through neural networks, thus eliminating the need for traditional mathematical programming solvers.

While inherently heuristic and not guaranteed to find the optimal or even a feasible solution, the framework often provides high-quality feasible solutions that are extremely useful either as alternatives to optimal solutions or as initial solutions for traditional solvers. This capability makes them invaluable tools in complex optimization scenarios where exact methods might struggle or be too slow.

## Features

- **Efficient Solution Acquisition**: The entire solution process relies entirely on neural networks without the need for mathematical programming solvers.

- **Integer Solution Guarantee**: Integrates learnable rounding directly into the network architecture, ensuring that solutions adhere strictly to integer constraints. 
