import numpy as np

def chord_method(func, a, b, epsilon, x_exact, max_iterations=None):
    fa, fb = func(a), func(b)
    if np.isnan(fa) or np.isnan(fb):
        raise ValueError("Funcția nu este definită la capetele intervalului.")

    if fa * fb >= 0:
        raise ValueError("Funcția trebuie să aibă valori de semne opuse la capetele intervalului.")

    try:
        q = abs((func(b) * (func(x_exact + epsilon) - 2 * func(x_exact) + func(x_exact - epsilon))) /
                 (x_exact ** 2 * (func(x_exact)) ** 2))
    except ZeroDivisionError:
        raise ValueError("Eroare la calculul lui q: împărțire la zero.")

    if max_iterations is None:
        max_iterations = int(np.log(abs(a - x_exact) / epsilon) / np.log(1 / q))

    iterations = 1
    xn = a  # Ensure xn is defined before the loop

    while abs(b - a) >= epsilon and iterations < max_iterations:
        xn = (a * fb - b * fa) / (fb - fa)  # Compute next approximation
        fxn = func(xn)

        if abs(fxn) < epsilon or abs(b - a) < epsilon:
            print(f"Iterations: {iterations}")  # Display the iteration count
            return xn, fxn, iterations

        a, b = b, xn
        fa, fb = fb, fxn

        iterations += 1

    return xn, func(xn), iterations  # Ensure xn is always returned
