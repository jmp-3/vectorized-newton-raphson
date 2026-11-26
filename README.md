# Vectorized Newton-Raphson Power Flow

This script solves the power flow of a 3-bus system with the Newton-Raphson method. It calculates the voltage magnitude and angle at each bus, and then the power at the slack bus and the generation at every bus.

The implementation is **vectorized and loopless**. Usual power flow codes use nested `for` loops over the buses to build the power injections and the Jacobian, element by element. This code has no loops over buses or matrix elements. It builds all terms at once with element-wise matrix operations and broadcasting (for example `V.*V'` and `theta - theta'`), and selects the diagonal and off-diagonal terms with mask matrices. The only loop is the Newton-Raphson iteration loop itself. The bus-level part of the code is therefore short, close to the written equations, and the same for any number of buses.

Developed for undergraduate coursework in dec/2020, later translated to numpy.

| File | Language |
|---|---|
| `fluxo.m` | GNU Octave, or MATLAB R2017a or newer |
| `fluxo.py` | Python 3 with NumPy and Matplotlib |

## How to run

**Octave or MATLAB:** go to the folder of the script and type `fluxo`. From a Linux terminal:

```bash
octave --persist fluxo.m
```

**Python:**

```bash
pip install numpy matplotlib
python3 fluxo.py
```

Both versions print the iterations and the final results, and open three figures with the convergence of $\Delta P$, $\Delta Q$ and the Jacobian elements. The code examples below use the Octave syntax. In the Python version, `sum(A)` is `np.sum(A, axis=0)`, `A'` is `A.T`, and the indices start at 0 (bus 2 is index 1).

## System

Base: 50 MVA, 120 kV. The lines have no resistance, so $G = 0$ and $Y = jB$.

| Bus | Type | Specified | Load (pu) |
|---|---|---|---|
| 1 | Slack | $V_1 = 1.0$, $\theta_1 = 0$ | $1 + j0.5$ |
| 2 | PQ | $P_2 = 1.5$, $Q_2 = 0.057$ | none |
| 3 | PQ | $P_3 = -1$, $Q_3 = -1$ | $1 + j1$ |

The unknowns of the problem are $\theta_2$, $\theta_3$, $V_2$ and $V_3$. The equations are the active and reactive power balances at buses 2 and 3.

## Method

### Power equations and Newton-Raphson step

With $\theta_{km} = \theta_k - \theta_m$:

$$
P_k = V_k \sum_{m} V_m \left( G_{km}\cos\theta_{km} + B_{km}\sin\theta_{km} \right), \qquad
Q_k = V_k \sum_{m} V_m \left( G_{km}\sin\theta_{km} - B_{km}\cos\theta_{km} \right)
$$

In each iteration the script calculates the mismatches $\Delta P = P^{\text{spec}} - P$ and $\Delta Q = Q^{\text{spec}} - Q$, and updates the unknowns:

$$
\begin{bmatrix} \theta \\ V \end{bmatrix}^{(i+1)} =
\begin{bmatrix} \theta \\ V \end{bmatrix}^{(i)} + J^{-1}
\begin{bmatrix} \Delta P \\ \Delta Q \end{bmatrix}
$$

It starts from $V = 1$ pu and $\theta = 0$ (flat start).

### Jacobian

For the formation of the Jacobian, the following formulas were implemented to define the matrices H, N, M and L:

| | $k = m$ | $k \neq m$ |
|---|---|---|
| $H = \partial P / \partial \theta$ | $-B_{kk}V_k^2 - Q_k$ | $V_k V_m (G_{km}\sin\theta_{km} - B_{km}\cos\theta_{km})$ |
| $N = \partial P / \partial V$ | $G_{kk}V_k + P_k/V_k$ | $V_k (G_{km}\cos\theta_{km} + B_{km}\sin\theta_{km})$ |
| $M = \partial Q / \partial \theta$ | $-G_{kk}V_k^2 + P_k$ | $-V_k V_m (G_{km}\cos\theta_{km} + B_{km}\sin\theta_{km})$ |
| $L = \partial Q / \partial V$ | $-B_{kk}V_k + Q_k/V_k$ | $V_k (G_{km}\sin\theta_{km} - B_{km}\cos\theta_{km})$ |

Following the unknowns of the problem, the Jacobian is reduced to:

$$
J =
\begin{bmatrix}
H_{22} & H_{23} & N_{22} & N_{23} \\
H_{32} & H_{33} & N_{32} & N_{33} \\
M_{22} & M_{23} & L_{22} & L_{23} \\
M_{32} & M_{33} & L_{32} & L_{33}
\end{bmatrix}
$$

### Implementation with matrix structures

The matrix structures were used to calculate the coefficients of the Jacobian without loops.

- `Vkl` is the matrix that results from the multiplication of the voltages that correspond to each of its indices. For example, `Vkl(1,2)` $= V_1 \times V_2$.
- `Vkk` has on its diagonal the squares of the voltages of its indices, for example `Vkk(2,2)` $= V_2 \times V_2$. The elements outside the diagonal are zero.
- `theta_kl = theta - theta'` contains the angle differences. Element $(k, m)$ of this matrix is $\theta_m - \theta_k$. The column sums `sum(...)` of this form give $P_k$ and $Q_k$ for each bus directly, so this form is used for the mismatches and for the diagonal terms.
- For the off-diagonal terms, element $(k, m)$ must use $\theta_k - \theta_m$. For this reason the code uses the transposed matrices `sin_km = sin_theta_kl'` and `cos_km = cos_theta_kl'`. The terms that multiply by $V_k$ use the column vector `V'`.

The formula for the positions $k = m$ is multiplied by a diagonal matrix, and the formula for the positions $k \neq m$ is multiplied by a matrix with zeros on the diagonal. For $n$ buses these structure matrices are:

```
Xkk = eye(n)        Xkm = ~eye(n)
   1   0   0           0   1   1
   0   1   0           1   0   1
   0   0   1           1   1   0
```

The matrix H is the sum of the two parts. N, M and L are formed in the same way:

$$
H =
\begin{bmatrix} H_{11} & 0 & 0 \\ 0 & H_{22} & 0 \\ 0 & 0 & H_{33} \end{bmatrix} +
\begin{bmatrix} 0 & H_{12} & H_{13} \\ H_{21} & 0 & H_{23} \\ H_{31} & H_{32} & 0 \end{bmatrix} =
\begin{bmatrix} H_{11} & H_{12} & H_{13} \\ H_{21} & H_{22} & H_{23} \\ H_{31} & H_{32} & H_{33} \end{bmatrix}
$$

This part of the code (between the comments `A partir daqui eh generico` and `A partir daqui nao eh mais generico`) works for any number of buses. The reduction of the Jacobian and the update of the unknowns are specific to this system.

## Results

| Bus | $V$ (pu) | $V$ (kV) | $\theta$ (deg) | $P_g$ (MW) | $Q_g$ (Mvar) |
|---|---|---|---|---|---|
| 1 | 1.0000 | 120.00 | 0 | 25.00 | 84.06 |
| 2 | 0.9189 | 110.27 | 5.1777 | 75.00 | 2.85 |
| 3 | 0.9020 | 108.24 | 1.1823 | 0 | 0 |

The largest mismatch goes from 1.5 to 0.19, 6.9e-3, 9.2e-6 and 1.6e-11 pu in the first 5 iterations. This is the quadratic convergence of Newton-Raphson.

Power balance: active generation (100 MW) is equal to the load (100 MW), because the lines have no losses. Reactive generation (86.91 Mvar) is 11.91 Mvar larger than the load (75 Mvar). This difference is the reactive power absorbed by the network, mainly by the line reactances.

## Limitations

- No PV buses and no reactive power limits.
