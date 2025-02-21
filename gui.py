import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import math
from methods.bisection import bisection_method
from methods.chord import chord_method
from methods.newton import newton_method
from methods.picard import picard_method
from utils import *

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
            "2*np.sin(3*x) - np.log(x**3 - 1) + 4",
            "np.sin(np.pi*x/6) - np.cos(x - 1)",
            "np.exp(-x) - x**3 + 8*np.cos(4*x)",
            "x**6 - 5.5*x**5 + 6.18*x**4 + 18.54*x**3 - 56.9592*x**2 + 55.9872*x - 19.3156",
            "x**6 - 0.7*x**5 - 8.7*x**4 + 5.58*x**3 + 22.356*x**2 - 8.39808*x - 19.3156",
            "x**6 - 2.4*x**5 - 18.27*x**4 + 23.216*x**3 + 115.7*x**2 - 19.5804*x - 164.818"
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
        self.entry_a.insert(0, "-1")
        self.entry_a.grid(row=0, column=1, padx=5)

        tk.Label(input_frame, text="b=").grid(row=0, column=2, padx=5)
        self.entry_b = tk.Entry(input_frame, width=10)
        self.entry_b.insert(0, "2")
        self.entry_b.grid(row=0, column=3, padx=5)

    def create_method_selection(self):
        frame = tk.Frame(self.main_frame, relief=tk.GROOVE, borderwidth=2)
        frame.pack(pady=5, fill="x", expand=True)

        tk.Label(frame, text="Precizarea metodei de calcul", font=("Arial", 12)).pack(pady=5)

        input_frame = tk.Frame(frame)
        input_frame.pack(pady=5)

        self.method_options = ["Selectați metoda", "Bisection", "Coardelor", "Newton", "Picard"]
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

    def find_zeros(self, func, a, b, epsilon):
        zeros = []
        x_vals = np.linspace(a, b, 100)
        y_vals = func(x_vals)

        # Find intervals where the function changes sign
        for i in range(len(x_vals) - 1):
            if y_vals[i] * y_vals[i + 1] < 0:
                x_exact = find_x_exact(func, x_vals[i], x_vals[i + 1])  # Get the exact point
                if self.method_dropdown.get() == "Bisection":
                    root, fx_value, iterations = bisection_method(func, x_vals[i], x_vals[i + 1], epsilon, x_exact)
                elif self.method_dropdown.get() == "Coardelor":
                    root, fx_value, iterations = chord_method(func, x_vals[i], x_vals[i + 1], epsilon,
                                                                   (x_vals[i] + x_vals[i + 1]) / 2,
                                                                   x_exact)  # Include x_exact
                elif self.method_dropdown.get() == "Newton":
                    root, fx_value, iterations = newton_method(func, x_vals[i], x_vals[i + 1], epsilon, x_exact)
                elif self.method_dropdown.get() == "Picard":
                    root, fx_value, iterations = picard_method(func, x_vals[i], x_vals[i + 1], epsilon)
                else:
                    continue  # Skip if no valid method is selected

                if root is not None:  # Check if root is valid
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

            # Use find_zeros to get the zeros based on the selected method
            zeros = self.find_zeros(func, a, b, epsilon)

            # If no specific method is selected, find zeros using find_x_exact
            if self.method_dropdown.get() == "Selectați metoda":
                x_vals = np.linspace(a, b, 100)
                y_vals = func(x_vals)
                for i in range(len(x_vals) - 1):
                    if y_vals[i] * y_vals[i + 1] < 0:
                        x_exact = find_x_exact(func, x_vals[i], x_vals[i + 1])
                        zeros.append((x_exact, func(x_exact), 0))  # 0 iterations for exact values

            # Prepare to plot the function
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
