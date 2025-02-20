import numpy as np

def picard_method(func, a, b, epsilon):
    # Define the contraction mapping Fi(x)
    Fi = lambda x: func(x)  # You may need to modify this based on your specific function

    # Step 1: Choose an initial guess x0
    x0 = (a + b) / 2  # Midpoint as initial guess
    x1 = Fi(x0)  # First iteration

    # Step 2: Check for conditions
    q = max(abs(Fi(x)) for x in np.linspace(a, b, 100))  # Calculate the maximum |Fi'(x)|

    if not (0 < q < 1):
        raise ValueError("The function is not a contraction mapping.")

    # Calculate maximum number of iterations
    Nmax = int(np.log(abs(x1 - x0) / epsilon) / np.log(1 / q))
    iterations = 0

    while iterations < Nmax:
        x0, x1 = x1, Fi(x1)  # Update x0 and compute the next approximation

        # Check stopping criteria
        if abs(x1 - x0) < epsilon:
            return x1, func(x1), iterations  # Return the root and function value

        iterations += 1

    return x1, func(x1), iterations  # Return the last computed value if Nmax reached