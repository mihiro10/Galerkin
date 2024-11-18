import approx_functions
import numpy as np

def main():
    try:
        N = int(input("Enter Nodes:  "))
        n = int(input("Enter Timesteps: "))
        xi = np.linspace(0, 1, N)
        h = xi[1] - xi[0]
        dt = 1 / n
        ts = np.linspace(0, 1, n + 1)

        approx_functions.solve_fem_problem(N, n, xi, ts, h, dt)
    except ValueError as ve:
        print("Invalid input. Please enter numeric values.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
