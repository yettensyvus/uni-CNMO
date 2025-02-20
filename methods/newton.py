import numpy as np

def newton_method(func, a, b, epsilon, x_exact):
    """Applies Newton's method to find a root of the function in the interval [a, b]."""
    try:
        # Define the first and second derivatives of the function
        f_prime = lambda x: (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon)
        f_double_prime = lambda x: (func(x + epsilon) - 2 * func(x) + func(x - epsilon)) / (epsilon ** 2)

        # Choose an initial guess x0 within [a, b]
        x0 = (a + b) / 2

        # Verify conditions for Newton's method
        if not (a <= x0 <= b):
            raise ValueError("Initial value x0 is not in the interval [a, b].")

        df_x0 = f_prime(x0)
        d2f_x0 = f_double_prime(x0)

        if abs(df_x0) < 1e-6:
            raise ValueError("f'(x0) is too small; the method may diverge.")

        if d2f_x0 is not None and df_x0 is not None:
            if d2f_x0 * df_x0 > 0:
                raise ValueError("The function does not satisfy the necessary conditions for convergence.")

        # Compute delta
        delta = abs(func(x_exact))

        # Compute maximum number of iterations
        log_arg = abs(x0 - x_exact) / epsilon
        if log_arg <= 0:
            raise ValueError("Invalid argument for logarithm in Nmax calculation.")

        Nmax = int(np.log2(np.log2(log_arg)))
        if Nmax <= 0:
            Nmax = 100  # Default maximum iterations if Nmax is invalid

        iterations = 0

        while iterations < Nmax:
            fx = func(x0)
            fpx = f_prime(x0)

            if abs(fx) < epsilon:
                return x0, fx, iterations

            if abs(fpx) < 1e-6:  # Prevent division by very small number
                raise ValueError("Derivative is too small, stopping iteration.")

            x1 = x0 - fx / fpx  # Newton's iteration step

            # Stopping criteria
            if abs(x1 - x0) < epsilon:
                return x1, func(x1), iterations

            if abs(func(x1)) < delta:
                return x1, func(x1), iterations

            x0 = x1
            iterations += 1

        return x0, func(x0), iterations  # Return last computed value if Nmax reached

    except Exception as e:
        print("Error in Newton's method:", e)
        return None, None, None