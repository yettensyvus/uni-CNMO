import numpy as np
from utils import check_continuity, check_uniqueness

def bisection_method(func, a, b, epsilon):
    # Step 1: Check if f(a) and f(b) are defined
    fa = func(a)
    fb = func(b)

    if np.isnan(fa) or np.isnan(fb):
        raise ValueError("Function is not defined at the endpoints a or b.")

    # Step 2: Check for continuity and sign change
    if not check_continuity(func, a, b):
        raise ValueError("Function is not continuous on the interval.")
    if fa * fb >= 0:
        raise ValueError("Function values at the endpoints must have opposite signs.")

    # Step 3: Check for uniqueness (strictly increasing or decreasing)
    if not check_uniqueness(func, a, b):
        raise ValueError("Function must be strictly increasing or decreasing on the interval.")

    # Calculate maximum number of iterations
    Nmax = int(np.ceil(np.log2((b - a) / epsilon)))
    iterations = 0

    while iterations < Nmax:
        # Step 4: Calculate the midpoint
        m = (a + b) / 2
        fm = func(m)

        # Check the stopping criteria
        if abs(b - a) < epsilon or abs(fm) < epsilon:
            return m, fm, iterations  # Return the midpoint as the approximate root

        # Step 5: Check the value of the function at m
        if fm == 0:
            return m, fm, iterations  # Found exact root
        elif fm * fb < 0:
            a = m  # Root is in [m, b]
            fa = fm  # Update fa
        else:
            b = m  # Root is in [a, m]
            fb = fm  # Update fb

        iterations += 1

    # Return the midpoint as the approximate root
    return (a + b) / 2, func((a + b) / 2), iterations