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
        self.entry_fx.insert(0, "x*np.cos(x)-np.sin(x)")
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

        self.method_options = ["Selectați metoda", "Bisection", "Secant"]
        self.method_dropdown = ttk.Combobox(input_frame, values=self.method_options, state="readonly")
        self.method_dropdown.current(0)
        self.method_dropdown.grid(row=0, column=0, padx=5)

        tk.Label(input_frame, text="eps=1e-").grid(row=0, column=1, padx=5)
        self.epsilon_spinbox = tk.Spinbox(input_frame, from_=1, to=10, width=5)
        self.epsilon_spinbox.insert(0, "6")
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

        # Calculate maximum number of iterations
        Nmax = int(np.ceil(np.log2((b - a) / epsilon)))
        iterations = 0

        while iterations < Nmax:
            # Step 3: Calculate the midpoint
            c = (a + b) / 2
            fc = func(c)

            # Calculate delta
            delta = abs(fc)

            # Check the stopping criteria
            if abs(b - a) < epsilon or delta < epsilon:
                return c, fc, iterations  # Return the midpoint as the approximate root

            # Step 4: Check the value of the function at c
            if fc == 0:
                return c, fc, iterations  # Found exact root
            elif fc * fb < 0:
                a = c  # Root is in [c, b]
                fa = fc  # Update fa
            else:
                b = c  # Root is in [a, c]
                fb = fc  # Update fb

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

    def check_uniqueness(self, func, a, b):
        # Check if the derivative maintains a consistent sign
        derivative = np.gradient([func(x) for x in np.linspace(a, b, 100)])
        return np.all(derivative > 0) or np.all(derivative < 0)

    def find_x_exact(self, func, a, b):
        x_vals = np.linspace(a, b, 400)
        y_vals = np.abs(func(x_vals))
        x_exact = x_vals[np.argmin(y_vals)]  # Punctul unde f(x) este cel mai aproape de 0
        return x_exact

    def chord_method(self, func, a, b, epsilon, delta, x_exact, max_iterations=None):
        # Verificăm dacă funcția este definită la capete
        fa, fb = func(a), func(b)
        if np.isnan(fa) or np.isnan(fb):
            raise ValueError("Funcția nu este definită la capetele intervalului.")

        # Verificăm continuitatea și schimbarea de semn
        if fa * fb >= 0:
            raise ValueError("Funcția trebuie să aibă valori de semne opuse la capetele intervalului.")

        # Calculăm q conform formulei
        try:
            q = abs((func(b) * (func(x_exact + epsilon) - 2 * func(x_exact) + func(x_exact - epsilon))) /
                    (x_exact ** 2 * (func(x_exact)) ** 2))
        except ZeroDivisionError:
            raise ValueError("Eroare la calculul lui q: împărțire la zero.")

        # Calculăm numărul maxim de iterații dacă nu este specificat
        if max_iterations is None:
            max_iterations = int(np.log(abs(a - x_exact) / epsilon) / np.log(1 / q))

        iterations = 0
        while abs(b - a) >= epsilon:
            # Calculăm punctul de intersecție a coardei
            xn = (a * fb - b * fa) / (fb - fa)
            fxn = func(xn)

            # Verificăm criteriile de oprire
            if abs(fxn) < delta or abs(b - a) < epsilon:
                return xn, fxn, iterations

            # Alegem noul interval
            if fxn * fb < 0:
                a, fa = xn, fxn
            else:
                b, fb = xn, fxn

            iterations += 1

            # Verificăm numărul maxim de iterații
            if iterations >= max_iterations:
                raise ValueError("Numărul maxim de iterații a fost atins fără convergență.")

        return xn, fxn, iterations

    def find_zeros(self, func, a, b, epsilon):
        zeros = []
        x_vals = np.linspace(a, b, 100)
        y_vals = func(x_vals)

        # Find intervals where the function changes sign
        for i in range(len(x_vals) - 1):
            if y_vals[i] * y_vals[i + 1] < 0:
                if self.method_dropdown.get() == "Bisection":
                    root, fx_value, iterations = self.bisection_method(func, x_vals[i], x_vals[i + 1], epsilon)
                elif self.method_dropdown.get() == "Secant":
                    root, fx_value, iterations = self.chord_method(func, x_vals[i], x_vals[i + 1], epsilon,
                                                                   (x_vals[i] + x_vals[i + 1]) / 2)
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
                raise ValueError("Function evaluation resulted in NaN values. Please check the function and the interval.")

            self.ax.clear()
            self.ax.plot(x_vals, y_vals, label=func_expr)

            for root, fx_value, _ in zeros:
                self.ax.scatter(root, fx_value, color="red", zorder=3)

            self.ax.axhline(0, color="black", linewidth=0.5)
            self.ax.legend()
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