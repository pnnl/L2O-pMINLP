## Quadratic Programming

$$
\begin{align*}
  \min \quad & \sum_{i=1}^n x_i^2 \\
  \text{subject to} \quad
  & x_i + x_{i+1} - p_i \geq 0 \quad \forall i \in \lbrace 1, \ldots, n-1 \rbrace \\
  & x_i + x_{i+1} - p_i \leq 5 \quad \forall i \in \lbrace 1, \ldots, n-1 \rbrace \\
  & x_n - x_1 - p_n \geq 0 \\
  & x_n - x_1 - p_n \leq 5 \\
  & x_i \in \mathbb{Z} \quad \forall i \in I \subseteq \lbrace 1, \ldots, n \rbrace
\end{align*}
$$

where $\mathbf{p} \in [1, 11]^n$.

## Rosenbrock Problem

$$
\begin{align*}
  \min \quad & \sum_{i=1}^n \left[ (1 - x_i)^2 + a_i (x_{i+1} - x_i^2)^2 \right] \\
  \text{subject to} \quad 
  & \sum_{i=0}^n (-1)^i x_i \geq 0 \\
  & \frac{p}{2} \leq \sum_{i=1}^n x_i^2 \leq p \\
  & x_i \in \mathbb{Z} \quad \forall i \in I \subseteq \lbrace 1, \ldots, n \rbrace
\end{align*}
$$

where $p \in [0.5, 6]$, and $\mathbf{a} \in [0.2, 1.2]^{n-1}$.

## Rastrigin Problem

$$
\begin{align*}
  \min \quad & \sum_{i=1}^n a_i + \sum_{i=1}^n \left[ x_i^2 - a_i \cdot \cos(2\pi x_i) \right] \\
  \text{subject to} \quad 
  & \sum_{i=1}^n (-1)^i x_i \geq 0 \\
  & \frac{p}{2} \leq \sum_{i=1}^n x_i^2 \leq p \\
  & -5.12 \leq x_i \leq 5.12 \quad \forall i \in \lbrace 1, \ldots, n \rbrace \\
  & x_i \in \mathbb{Z} \quad \forall i \in I \subseteq \lbrace 1, \ldots, n \rbrace
\end{align*}
$$

where $p \in [2, 6]$, and $a \in [6, 15]$.
