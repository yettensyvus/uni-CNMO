import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
import math

class FunctionSelectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Metode numerice de calcul al rădăcinilor ecuațiilor neliniare")
        self.center_window(900, 600)

        # Main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(expand=True, pady=10)

        # UI Elements
        self.create_title_label()
        self.create_function_selection()
        self.create_interval_selection()
        self.create_method_selection()
        self.create_execute_button()
        self.create_graph_and_table()

    def center_window(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x_offset = (screen_width - width) // 2
        y_offset = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x_offset}+{y_offset}")

    def create_title_label(self):
        label = tk.Label(self.main_frame, text="Metode numerice de calcul al rădăcinilor ecuațiilor neliniare",
                         font=("Arial", 16), anchor="center")
        label.pack(pady=10)

    def create_function_selection(self):
        frame = tk.Frame(self.main_frame, relief=tk.GROOVE, borderwidth=2)
        frame.pack(pady=5, fill="x", expand=True)

        tk.Label(frame, text="Precizarea membrului stâng al ecuației", font=("Arial", 12)).pack(pady=5)

        radio_frame = tk.Frame(frame)
        radio_frame.pack(pady=5)

        self.selected_option = tk.StringVar(value="interactiv")

        tk.Radiobutton(radio_frame, text="din colecție", variable=self.selected_option, value="din_colectie").pack(
            side=tk.LEFT, padx=20)
        tk.Radiobutton(radio_frame, text="interactiv", variable=self.selected_option, value="interactiv").pack(
            side=tk.RIGHT, padx=20)

        input_frame = tk.Frame(frame)
        input_frame.pack(pady=5)

        tk.Label(input_frame, text="f(x) =").grid(row=0, column=0, padx=5)

        # New function collection
        self.function_options = [
            "2*np.sin(3*x) - np.log(x**2 - 1) + 4",
            "np.sin(np.pi*x/6) - np.cos(x-1)",
            "np.exp(-x**2) + 8*np.cos(4*x)"
        ]
        self.dropdown = ttk.Combobox(input_frame, values=self.function_options, state="readonly")
        self.dropdown.current(0)
        self.dropdown.grid(row=0, column=1, padx=5)

        tk.Label(input_frame, text="f(x) =").grid(row=0, column=2, padx=5)
        self.entry_fx = tk.Entry(input_frame, width=20)
        self.entry_fx.insert(0, "x**2 - np.exp(x)")
        self.entry_fx.grid(row=0, column=3, padx=5)

    def create_interval_selection(self):
        frame = tk.Frame(self.main_frame, relief=tk.GROOVE, borderwidth=2)
        frame.pack(pady=5, fill="x", expand=True)

        tk.Label(frame, text="Intervalul de căutare a zerourilor", font=("Arial", 12)).pack(pady=5)

        input_frame = tk.Frame(frame)
        input_frame.pack(pady=5)

        tk.Label(input_frame, text="a=").grid(row=0, column=0, padx=5)
        self.entry_a = tk.Entry(input_frame, width=10)
        self.entry_a.insert(0, "-9")
        self.entry_a.grid(row=0, column=1, padx=5)

        tk.Label(input_frame, text="b=").grid(row=0, column=2, padx=5)
        self.entry_b = tk.Entry(input_frame, width=10)
        self.entry_b.insert(0, "8")
        self.entry_b.grid(row=0, column=3, padx=5)

    def create_method_selection(self):
        frame = tk.Frame(self.main_frame, relief=tk.GROOVE, borderwidth=2)
        frame.pack(pady=5, fill="x", expand=True)

        tk.Label(frame, text="Precizarea metodei de calcul", font=("Arial", 12)).pack(pady=5)

        input_frame = tk.Frame(frame)
        input_frame.pack(pady=5)

        self.method_options = ["Selectați metoda", "Bisection", "Secant", "Newton"]
        self.method_dropdown = ttk.Combobox(input_frame, values=self.method_options, state="readonly")
        self.method_dropdown.current(0)
        self.method_dropdown.grid(row=0, column=0, padx=5)

        tk.Label(input_frame, text="eps=1e-").grid(row=0, column=1, padx=5)
        self.epsilon_spinbox = tk.Spinbox(input_frame, from_=1, to=10, width=5, textvariable=tk.StringVar(value="4"))
        self.epsilon_spinbox.grid(row=0, column=2, padx=5)

    def create_execute_button(self):
        execute_button = tk.Button(self.main_frame, text="Execută!", font=("Arial", 12), command=self.plot_function)
        execute_button.pack(pady=10)

    def create_graph_and_table(self):
        frame = tk.Frame(self.main_frame)
        frame.pack(pady=10, fill="both", expand=True)

        graph_frame = tk.Frame(frame)
        graph_frame.pack(side=tk.LEFT, padx=10, pady=5, fill="both", expand=True)

        self.figure, self.ax = plt.subplots(figsize=(6.25, 3))
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        table_frame = tk.Frame(frame)
        table_frame.pack(side=tk.RIGHT, padx=10, pady=5, fill="both", expand=True)

        self.tree = ttk.Treeview(table_frame, columns=("x", "f(x)", "Iterations", "Time"), show="headings")
        self.tree.heading("x", text="Rădăcina x")
        self.tree.heading("f(x)", text="f(x)")
        self.tree.heading("Iterations", text="Număr iterații k")
        self.tree.heading("Time", text="Timp de calcul")
        self.tree.pack(fill="both", expand=True)

    def bisection_method(self, func, a, b, epsilon):
        # Step 1: Check if f(a) and f(b) are defined
        fa = func(a)
        fb = func(b)

        if np.isnan(fa) or np.isnan(fb):
            raise ValueError("Function is not defined at the endpoints a or b.")

        # Step 2: Check for continuity and sign change
        if not self.check_continuity(func, a, b):
            raise ValueError("Function is not continuous on the interval.")
        if fa * fb >= 0:
            raise ValueError("Function values at the endpoints must have opposite signs.")

        # Step 3: Check for uniqueness (strictly increasing or decreasing)
        if not self.check_uniqueness(func, a, b):
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

    def check_continuity(self, func, a, b, num_points=100):
        x_vals = np.linspace(a, b, num_points)
        try:
            func(x_vals)
            return True
        except Exception:
            return False

    def check_uniqueness(self, func, a, b, num_points=100):
        x_vals = np.linspace(a, b, num_points)
        y_vals = np.gradient([func(x) for x in x_vals])
        return np.all(y_vals > 0) or np.all(y_vals < 0)  # Check if the function is strictly increasing or decreasing

    def find_x_exact(self, func, a, b):
        x_vals = np.linspace(a, b, 400)
        y_vals = np.abs(func(x_vals))
        x_exact = x_vals[np.argmin(y_vals)]  # Punctul unde f(x) este cel mai aproape de 0
        return x_exact

    def chord_method(self, func, a, b, epsilon, x_exact, max_iterations=None):
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

        iterations = 0
        xn = a  # Ensure xn is defined before the loop

        while abs(b - a) >= epsilon and iterations < max_iterations:
            xn = (a * fb - b * fa) / (fb - fa)  # Compute next approximation
            fxn = func(xn)

            if abs(fxn) < epsilon or abs(b - a) < epsilon:
                return xn, fxn, iterations

            a, b = b, xn
            fa, fb = fb, fxn
            iterations += 1

        return xn, func(xn), iterations  # Ensure xn is always returned

    def newton_method(self, func, a, b, epsilon, x_exact):
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

    def find_zeros(self, func, a, b, epsilon):
        zeros = []
        x_vals = np.linspace(a, b, 100)
        y_vals = func(x_vals)

        # Find intervals where the function changes sign
        for i in range(len(x_vals) - 1):
            if y_vals[i] * y_vals[i + 1] < 0:
                x_exact = self.find_x_exact(func, x_vals[i], x_vals[i + 1])  # Get the exact point
                if self.method_dropdown.get() == "Bisection":
                    root, fx_value, iterations = self.bisection_method(func, x_vals[i], x_vals[i + 1], epsilon)
                elif self.method_dropdown.get() == "Secant":
                    root, fx_value, iterations = self.chord_method(func, x_vals[i], x_vals[i + 1], epsilon,
                                                                   (x_vals[i] + x_vals[i + 1]) / 2,
                                                                   x_exact)  # Include x_exact
                elif self.method_dropdown.get() == "Newton":
                    root, fx_value, iterations = self.newton_method(func, x_vals[i], x_vals[i + 1], epsilon, x_exact)
                zeros.append((root, fx_value, iterations))
        return zeros

    def plot_function(self):
        try:
            # Determine the function based on the selected option
            if self.selected_option.get() == "din_colectie":
                func_expr = self.dropdown.get()
            else:
                func_expr = self.entry_fx.get()

            func = lambda x: eval(func_expr, {"x": x, "np": np, "math": math})

            a = float(self.entry_a.get())
            b = float(self.entry_b.get())
            epsilon = 10 ** -int(self.epsilon_spinbox.get())

            self.tree.delete(*self.tree.get_children())

            zeros = self.find_zeros(func, a, b, epsilon)

            x_vals = np.linspace(a, b, 400)
            y_vals = func(x_vals)

            # Check for NaN values and handle them
            if np.any(np.isnan(y_vals)):
                raise ValueError(
                    "Function evaluation resulted in NaN values. Please check the function and the interval.")

            self.ax.clear()
            self.ax.plot(x_vals, y_vals, label=func_expr)

            for root, fx_value, _ in zeros:
                self.ax.scatter(root, fx_value, color="red", zorder=3)

            # Calculate the maximum root from the found zeros
            if zeros:
                max_root = max(zeros, key=lambda item: abs(item[0]))[0]  # Get the root with maximum absolute value
                self.ax.scatter(max_root, func(max_root), color="green", zorder=4, label=f'Max Root: {max_root:.4f}')

            self.ax.axhline(0, color="black", linewidth=0.5)

            # Set the legend to be at the top center
            self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)

            self.canvas.draw()

            for root, fx_value, iterations in zeros:
                start_time = time.time()
                self.tree.insert("", "end", values=(
                    round(root, 4), round(fx_value, 4), iterations, round(time.time() - start_time, 6)))

            self.tree.column("x", width=100, anchor="center")
            self.tree.column("f(x)", width=100, anchor="center")
            self.tree.column("Iterations", width=100, anchor="center")
            self.tree.column("Time", width=100, anchor="center")

        except Exception as e:
            print("Error in function evaluation:", e)


if __name__ == "__main__":
    root = tk.Tk()
    app = FunctionSelectionApp(root)
    root.mainloop()