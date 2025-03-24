import numpy as np

def picard_method(func, a, b, epsilon):
    # Initial guess
    x_n = (a + b) / 2  # Midpoint as initial value


    # Convergence check: estimate q using numerical derivative
    q = max(abs((func(x + 1e-5) - func(x)) / 1e-5) for x in np.linspace(a, b, 100))

    #print(f"Computed contraction coefficient q: {q}")  # Debugging information

    # Maximum number of iterations
    Nmax = int(np.log(epsilon) / np.log(q))

    iterations = 1

    while iterations < Nmax:
        x_next = func(x_n)

        # Check stopping criteria
        if abs(x_next - x_n) < epsilon:
            return x_next, func(x_next), iterations + 1

        x_n = x_next
        iterations += 1

    return x_n, func(x_n), iterations  # Return the last computed value if Nmax reached
