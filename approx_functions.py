import numpy as np
import matplotlib.pyplot as plt

# Basic mathematical functions
def calculate_source_term(x, time):
    """
    Calculate the source term f(x,t) for the differential equation.
    Args:
        x (float/array): Spatial coordinate(s)
        time (float): Time value
    Returns:
        float/array: Value of source term
    """
    return (np.pi**2 - 1) * np.exp(-time) * np.sin(np.pi * x)

def calculate_exact_solution(x, time):
    """
    Calculate the analytical solution u(x,t) to the differential equation.
    Args:
        x (float/array): Spatial coordinate(s)
        time (float): Time value
    Returns:
        float/array: Exact solution value
    """
    return np.exp(-time) * np.sin(np.pi * x)

def calculate_initial_condition(x):
    """
    Calculate the initial condition u(x,0) for the problem.
    Args:
        x (float/array): Spatial coordinate(s)
    Returns:
        float/array: Initial condition value
    """
    return np.sin(np.pi * x)

# Matrix initialization and utilities
def create_empty_matrices(node_count, time_steps):
    """
    Initialize empty matrices needed for FEM computation.
    Args:
        node_count (int): Number of spatial nodes
        time_steps (int): Number of time steps
    Returns:
        tuple: (mass_matrix, stiffness_matrix, force_matrix)
    """
    return (np.zeros((node_count, node_count)), 
            np.zeros((node_count, node_count)), 
            np.zeros((node_count, time_steps + 1)))

def create_element_mapping(node_count):
    """
    Create mapping between local and global node indices.
    Args:
        node_count (int): Number of nodes
    Returns:
        ndarray: Array mapping local element nodes to global nodes
    """
    return np.vstack((np.arange(0, node_count - 1), np.arange(1, node_count))).T

# FEM core functions
def create_basis_functions(element_length):
    """
    Generate linear basis functions and their derivatives for FEM.
    Args:
        element_length (float): Length of finite element
    Returns:
        tuple: (basis_functions_at_quadrature, derivatives, scaling_factors)
    """
    phi1 = lambda zeta: (1 - zeta) / 2
    phi2 = lambda zeta: (1 + zeta) / 2
    basis_function_derivatives = np.array([-1 / 2, 1 / 2])
    quadrature_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    basis_functions_at_quad = np.array([[phi1(zeta), phi2(zeta)] for zeta in quadrature_points])
    return (basis_functions_at_quad, basis_function_derivatives, 2 / element_length, element_length / 2)

def assemble_fem_matrices(node_count, time_steps, stiffness_matrix, mass_matrix, force_matrix, mapping, basis_functions, derivatives, derivative_scaling, integral_scaling, element_length):
    """
    Assemble global FEM matrices from local element contributions.
    Args:
        node_count (int): Number of nodes
        time_steps (int): Number of time steps
        stiffness_matrix (ndarray): Global stiffness matrix
        mass_matrix (ndarray): Global mass matrix
        force_matrix (ndarray): Global force vector
        mapping (ndarray): Element to node mapping
        basis_functions (ndarray): Basis functions at quadrature points
        derivatives (ndarray): Basis function derivatives
        derivative_scaling (float): Scaling factor for derivatives
        integral_scaling (float): Scaling factor for integration
        element_length (float): Length of finite element
    Returns:
        tuple: (mass_matrix, stiffness_matrix, force_matrix)
    """
    quadrature_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    for element_index in range(node_count - 1):
        local_mass_matrix = np.zeros((2, 2))
        local_stiffness_matrix = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                local_mass_matrix[i, j] = sum(basis_functions[i, k] * basis_functions[j, k] for k in range(2)) * element_length
                local_stiffness_matrix[i, j] = derivatives[i] * derivative_scaling * derivatives[j] * derivative_scaling * integral_scaling * 2
        global_nodes = mapping[element_index].astype(int)
        for i in range(2):
            for j in range(2):
                stiffness_matrix[global_nodes[i], global_nodes[j]] += local_stiffness_matrix[i, j]
                mass_matrix[global_nodes[i], global_nodes[j]] += local_mass_matrix[i, j]
        force_matrix[element_index, :] = -sum(calculate_source_term(zeta, time_steps) * basis_functions[0, k] for k, zeta in enumerate(quadrature_points)) * (1 / 8)
    return mass_matrix, stiffness_matrix, force_matrix

def apply_dirichlet_conditions(mass_matrix, node_count):
    """
    Apply Dirichlet boundary conditions to system matrices.
    Args:
        mass_matrix (ndarray): Mass matrix to modify
        node_count (int): Number of nodes
    Returns:
        tuple: (modified_mass_matrix, boundary_conditions_matrix)
    """
    mass_matrix[0, :] = mass_matrix[-1, :] = mass_matrix[:, 0] = mass_matrix[:, -1] = 0
    mass_matrix[0, 0] = mass_matrix[-1, -1] = 1
    dirichlet_boundary_conditions = np.eye(node_count)
    dirichlet_boundary_conditions[0, 0] = dirichlet_boundary_conditions[-1, -1] = 0
    return mass_matrix, dirichlet_boundary_conditions

# Time integration functions
def setup_euler_matrices(mass_matrix, stiffness_matrix, time_step):
    """
    Prepare matrices needed for Euler time integration methods.
    Args:
        mass_matrix (ndarray): Mass matrix
        stiffness_matrix (ndarray): Stiffness matrix
        time_step (float): Time step size
    Returns:
        tuple: (inverse_mass_matrix, mass_stiffness_product, inverse_euler_matrix)
    """
    inverse_mass_matrix = np.linalg.inv(mass_matrix)
    euler_matrix = (1 / time_step) * mass_matrix + stiffness_matrix
    return inverse_mass_matrix, np.dot(inverse_mass_matrix, stiffness_matrix), np.linalg.inv(euler_matrix)

def solve_euler_timesteps(node_count, time_steps, time_step, mass_stiffness_product, inverse_mass_matrix, mass_matrix, force_matrix, boundary_conditions, euler_method, nodes, inverse_euler_matrix):
    """
    Solve the time-dependent problem using Forward or Backward Euler method.
    Args:
        node_count (int): Number of nodes
        time_steps (int): Number of time steps
        time_step (float): Time step size
        mass_stiffness_product (ndarray): Product of inverse mass matrix and stiffness matrix
        inverse_mass_matrix (ndarray): Inverse of mass matrix
        mass_matrix (ndarray): Mass matrix
        force_matrix (ndarray): Force vector
        boundary_conditions (ndarray): Boundary conditions matrix
        euler_method (str): 'FE' for Forward Euler or 'BE' for Backward Euler
        nodes (ndarray): Spatial node coordinates
        inverse_euler_matrix (ndarray): Inverse of Euler matrix for BE method
    Returns:
        ndarray: Solution matrix for all time steps
    """
    solution = np.zeros((node_count, time_steps + 1))
    solution[:, 0] = calculate_initial_condition(nodes)
    for t in range(time_steps):
        if euler_method == 'FE':
            solution[:, t + 1] = solution[:, t] - time_step * mass_stiffness_product.dot(solution[:, t]) + time_step * inverse_mass_matrix.dot(force_matrix[:, t])
        else:
            solution[:, t + 1] = (1 / time_step) * inverse_euler_matrix.dot(mass_matrix.dot(solution[:, t])) + inverse_euler_matrix.dot(force_matrix[:, t])
        solution[:, t + 1] = boundary_conditions.dot(solution[:, t + 1])
    return solution

# Visualization
def plot_comparison(x_continuous, analytical_solution, x_discrete, numerical_solution, time_step_count, euler_method):
    """
    Plot comparison between analytical and numerical solutions.
    Args:
        x_continuous (ndarray): Points for plotting analytical solution
        analytical_solution (ndarray): Analytical solution values
        x_discrete (ndarray): FEM node points
        numerical_solution (ndarray): Numerical solution values
        time_step_count (int): Number of time steps
        euler_method (str): 'FE' or 'BE' to indicate method used
    """
    plt.plot(x_continuous, analytical_solution, label='True Function', color="blue")
    method_label = "Forward Euler Approximation" if euler_method == "FE" else "Backward Euler Approximation"
    plt.plot(x_discrete, numerical_solution[:, time_step_count], label=f'{method_label} with n = {time_step_count}', color="red")
    plt.xlabel('x')
    plt.ylabel('Solution')
    plt.title('True Function vs FEM Approximations')
    plt.legend()
    plt.show()

def solve_fem_problem(N, n, xi, ts, h, dt):
    """
    Main function to solve the FEM problem with user-selected Euler method.
    Args:
        N (int): Number of spatial nodes
        n (int): Number of time steps
        xi (ndarray): Spatial node coordinates
        ts (float): Total simulation time
        h (float): Spatial step size
        dt (float): Time step size
    """
    while True:
        method = input("Choose between forward euler or backward euler\nType FE or BE: ").upper()
        if method in ['FE', 'BE']:
            mass_matrix, stiffness_matrix, force_matrix = create_empty_matrices(N, n)
            mapping = create_element_mapping(N)
            basis_functions, derivatives, derivative_scaling, integral_scaling = create_basis_functions(h)
            mass_matrix, stiffness_matrix, force_matrix = assemble_fem_matrices(N, ts, stiffness_matrix, mass_matrix, force_matrix, mapping, basis_functions, derivatives, derivative_scaling, integral_scaling, h)
            mass_matrix, dirichlet_bc = apply_dirichlet_conditions(mass_matrix, N)
            inverse_mass_matrix, mass_stiffness_product, inverse_euler_matrix = setup_euler_matrices(mass_matrix, stiffness_matrix, dt)
            solution = solve_euler_timesteps(N, n, dt, mass_stiffness_product, inverse_mass_matrix, mass_matrix, force_matrix, dirichlet_bc, method, xi, inverse_euler_matrix)
            x = np.linspace(0, 1, N)
            xn = np.linspace(0, 1, 1000)
            sol = calculate_exact_solution(xn, 1)
            plot_comparison(xn, sol, x, solution, n, method)
            break
        else:
            print("Error: Did not Pick FE or BE")