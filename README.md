# Heat Transfer Problem - 1D Galerkin Method

## Project Overview

This project involves solving a one-dimensional heat transfer problem using the Galerkin finite element method. The problem considers the equation:

$$u_t - u_{xx} = f(x, t), \quad (x, t) \in (0, 1) \times (0, 1)$$

with initial and Dirichlet boundary conditions:

- **Initial Condition**:  
  $$u(x, 0) = \sin(\pi x)$$

- **Boundary Conditions**:  
  $$u(0, t) = u(1, t) = 0$$

- **Source Function**:  
  $$f(x, t) = (\pi^2 - 1)e^{-t} \sin(\pi x)$$

The analytic solution for this problem is:

$$u(x, t) = e^{-t} \sin(\pi x)$$

The solution is implemented using a 1D Galerkin finite element approach with **N = 11 nodes**, but the code is designed to handle a generalized number of nodes. The 

## Usage
1. **Code Setup**: Clone the repository and run the script using Python.
2. **Running Script**: Run 'python solve.py' . 'approx_functions.py' is the script that holds the functions for calculations and 'solve.py' holds the script that runs the functions.
3. **Modify Parameters**: The number of nodes (N) and the time step (dt) can be adjusted within the code to observe the changes in stability and accuracy.
4. **Plot Results**: The solution is plotted at the final time step, and the plots can be used to analyze the performance of different schemes.


## Features

1. **Generalized Node Count**: The code supports different numbers of nodes, allowing for greater flexibility in mesh refinement.
2. **Initial and Boundary Condition Handling**: The boundary and initial conditions are implemented to ensure the correct setup for this specific problem.
3. **Forward and Backward Euler Time Discretization**: Both explicit (forward) and implicit (backward) Euler time-stepping schemes are included to observe their stability behavior.
4. **Elemental Mass and Stiffness Matrices**: The elemental matrices are computed by mapping from the physical domain (x-space) to the reference domain (xi-space), integrated in the parent space, and then assembled into a global system.

## Requirements
- Python 3 or similar language with numerical computation capabilities.
- Libraries such as NumPy and Matplotlib for matrix operations and visualization.

## Implementation Steps

### 1. Weak Form Derivation
The weak form of the equation is derived by hand and is included in the repository. This involves integrating against a test function and applying integration by parts.

### 2. Explicit Forward Euler Scheme
- **Time Discretization**: Use a time step of $$\Delta t = \frac{1}{551}$$.
- **Stability Investigation**: Increase the time step until instability is observed and record the value at which this occurs. The instability occurs at $$\Delta t = \frac{1}{277}$$, as shown in the project analysis.
- **Node Count Effect**: As the number of nodes (N) decreases, the numerical approximation undershoots the true solution. This effect can be seen in the results when using $$N = 5$$.

### 3. Implicit Backward Euler Scheme
- **Time Discretization**: Use the same time steps as the forward Euler scheme.
- **Stability Observation**: When using the backward Euler method at a time step of $$\Delta t = \frac{1}{277}$$, the solution remains stable. This is due to the unconditional stability of the backward Euler method. However, if the time step becomes equal to or greater than the spatial step size, the solution becomes less accurate due to rough approximations of the derivative, as seen in the project analysis. This is seen in the figure where time step = 11



## Results
- The **Forward Euler** method demonstrates instability at larger time steps, with instability observed at $$\Delta t = \frac{1}{277}$$.
- The **Backward Euler** method remains stable even for larger time steps, highlighting its unconditional stability.
- The code allows exploration of the effect of node count and time step size on the stability and accuracy of the solution. As the time step increases beyond the spatial step size, the approximation of the derivative becomes less accurate, leading to a decrease in solution accuracy.
