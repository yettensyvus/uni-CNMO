import numpy as np

def check_continuity(func, a, b, num_points=100):
    x_vals = np.linspace(a, b, num_points)
    try:
        func(x_vals)
        return True
    except Exception:
        return False

def check_uniqueness(func, a, b, num_points=100):
    x_vals = np.linspace(a, b, num_points)
    y_vals = np.gradient([func(x) for x in x_vals])
    return np.all(y_vals > 0) or np.all(y_vals < 0)  # Check if the function is strictly increasing or decreasing

def find_x_exact(func, a, b):
    x_vals = np.linspace(a, b, 400)
    y_vals = np.abs(func(x_vals))
    x_exact = x_vals[np.argmin(y_vals)]  # Punctul unde f(x) este cel mai aproape de 0
    return x_exact