import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.interpolate import interp1d, CubicSpline

class FunctionSelectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aproximarea funcțiilor prin interpolare")
        self.center_window(1200, 600)

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(expand=True, pady=10)

        self.create_title_label()
        self.create_function_selection()
        self.create_interval_selection()
        self.create_method_and_network_selection()
        self.create_execute_button()
        self.create_graphs()

    def center_window(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x_offset = (screen_width - width) // 2
        y_offset = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x_offset}+{y_offset}")

    def create_title_label(self):
        label = tk.Label(self.main_frame, text="Aproximarea funcțiilor prin interpolare",
                         font=("Arial", 16), anchor="center")
        label.pack(pady=10)

    def create_function_selection(self):
        frame = tk.Frame(self.main_frame, relief=tk.GROOVE, borderwidth=2)
        frame.pack(pady=5, fill="x", expand=True)

        tk.Label(frame, text="Precizarea funcției", font=("Arial", 12)).pack(pady=5)

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
        self.function_options = [
            "x * np.exp(2 * np.sin(x))",
            "np.sin(x) + np.cos(x)",
            "np.sin(np.pi * x / 6) - np.cos(x - 1)",
            "np.exp(-x) - x**3 + 8 * np.cos(4 * x)",
            "x**3 - 5 * np.arctan(x)"
        ]
        self.dropdown = ttk.Combobox(input_frame, values=self.function_options, state="readonly")
        self.dropdown.current(0)
        self.dropdown.grid(row=0, column=1, padx=5)

        tk.Label(input_frame, text="f(x) =").grid(row=0, column=2, padx=5)
        self.entry_fx = tk.Entry(input_frame, width=20)
        self.entry_fx.insert(0, "x * np.exp(2 * np.sin(x))")
        self.entry_fx.grid(row=0, column=3, padx=5)

    def create_interval_selection(self):
        frame = tk.Frame(self.main_frame, relief=tk.GROOVE, borderwidth=2)
        frame.pack(pady=5, fill="x", expand=True)

        tk.Label(frame, text="Intervalul de aproximare", font=("Arial", 12)).pack(pady=5)

        input_frame = tk.Frame(frame)
        input_frame.pack(pady=5)

        tk.Label(input_frame, text="a=").grid(row=0, column=0, padx=5)
        self.entry_a = tk.Entry(input_frame, width=10)
        self.entry_a.insert(0, "-10")
        self.entry_a.grid(row=0, column=1, padx=5)

        tk.Label(input_frame, text="b=").grid(row=0, column=2, padx=5)
        self.entry_b = tk.Entry(input_frame, width=10)
        self.entry_b.insert(0, "10")
        self.entry_b.grid(row=0, column=3, padx=5)

    def create_method_and_network_selection(self):
        frame = tk.Frame(self.main_frame, relief=tk.GROOVE, borderwidth=2)
        frame.pack(pady=5, fill="x", expand=True)

        tk.Label(frame, text="Metoda și tipul rețelei", font=("Arial", 12)).pack(pady=5)

        input_frame = tk.Frame(frame)
        input_frame.pack(pady=5)

        tk.Label(input_frame, text="Metoda:").grid(row=0, column=0, padx=5)
        self.method_options = ["Lagrange", "Lagrange pe porțiuni", "Spline cubic"]
        self.method_dropdown = ttk.Combobox(input_frame, values=self.method_options, state="readonly")
        self.method_dropdown.current(0)
        self.method_dropdown.grid(row=0, column=1, padx=5)

        tk.Label(input_frame, text="Tipul rețelei:").grid(row=0, column=2, padx=5)
        self.network_options = ["Echidistantă", "Chebyshev", "Aleatoare"]
        self.network_dropdown = ttk.Combobox(input_frame, values=self.network_options, state="readonly")
        self.network_dropdown.current(0)
        self.network_dropdown.grid(row=0, column=3, padx=5)

        tk.Label(input_frame, text="n =").grid(row=0, column=4, padx=5)
        self.node_spinbox = tk.Spinbox(input_frame, from_=1, to=100, width=5, textvariable=tk.StringVar(value="4"))
        self.node_spinbox.grid(row=0, column=5, padx=5)

    def create_execute_button(self):
        execute_button = tk.Button(self.main_frame, text="Execută!", font=("Arial", 12), command=self.plot_graphs)
        execute_button.pack(pady=10)

    def create_graphs(self):
        graph_frame = tk.Frame(self.main_frame)
        graph_frame.pack(fill="both", expand=True)

        figure_size = (5, 4)  # Width, Height in inches

        # Create function plot (column 1)
        self.figure1, self.ax1 = plt.subplots(figsize=figure_size)
        self.canvas1 = FigureCanvasTkAgg(self.figure1, master=graph_frame)
        self.canvas1.get_tk_widget().grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        # Create error plot (column 2)
        self.figure2, self.ax2 = plt.subplots(figsize=figure_size)
        self.canvas2 = FigureCanvasTkAgg(self.figure2, master=graph_frame)
        self.canvas2.get_tk_widget().grid(row=0, column=1, padx=10, pady=5, sticky="nsew")

        # Configure grid weights to ensure proper resizing
        graph_frame.grid_rowconfigure(0, weight=1)
        graph_frame.grid_columnconfigure(0, weight=1)  # Function plot
        graph_frame.grid_columnconfigure(1, weight=1)  # Error plot

    def lagrange_interpolation(self, x, x_nodes, y_nodes):
        n = len(x_nodes)
        result = np.zeros_like(x, dtype=float)
        for i in range(n):
            basis = np.ones_like(x, dtype=float)
            for j in range(n):
                if i != j:
                    basis *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
            result += y_nodes[i] * basis
        return result

    def piecewise_lagrange(self, x, x_nodes, y_nodes, degree=1):
        interp_func = interp1d(x_nodes, y_nodes, kind='linear' if degree == 1 else 'quadratic',
                              bounds_error=False, fill_value="extrapolate")
        return interp_func(x)

    def plot_graphs(self):
        self.ax1.clear()
        self.ax2.clear()

        try:
            a = float(self.entry_a.get())
            b = float(self.entry_b.get())
            n = int(self.node_spinbox.get())
            method = self.method_dropdown.get()
            network = self.network_dropdown.get()

            if self.selected_option.get() == "din_colectie":
                func_str = self.function_options[self.dropdown.current()]
            else:
                func_str = self.entry_fx.get()

            def f(x):
                return eval(func_str)

            if network == "Echidistantă":
                x_nodes = np.linspace(a, b, n)
            elif network == "Chebyshev":
                x_nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(np.linspace(np.pi, 0, n))
            else:  # Aleatoare
                x_nodes = np.sort(np.random.uniform(a, b, n))

            y_nodes = f(x_nodes)
            x_fine = np.linspace(a, b, 200)  # Fixed number of points for fine grid
            y_fine = f(x_fine)

            if method == "Lagrange":
                y_interp = self.lagrange_interpolation(x_fine, x_nodes, y_nodes)
            elif method == "Lagrange pe porțiuni":
                y_interp = self.piecewise_lagrange(x_fine, x_nodes, y_nodes, degree=1)
            elif method == "Spline cubic":
                cs = CubicSpline(x_nodes, y_nodes)
                y_interp = cs(x_fine)

            error = y_fine - y_interp
            error_nodes = np.zeros(n)

            self.ax1.plot(x_fine, y_fine, 'b-', label="Funcția f(x)")
            self.ax1.plot(x_fine, y_interp, 'g--', label="Interpolare")
            self.ax1.plot(x_nodes, y_nodes, 'ro', label="Noduri")
            self.ax1.set_title(f"Metoda {method} ({network})")
            self.ax1.legend()
            self.ax1.grid(True)

            self.ax2.plot(x_fine, error, 'r-', label="Eroare f(x) - p(x)")
            self.ax2.plot(x_nodes, error_nodes, 'bo', label="Noduri")
            self.ax2.plot(x_nodes, error_nodes, 'b--', label="Linia nodurilor")
            self.ax2.set_title("Graficul funcției erorii")
            self.ax2.legend()
            self.ax2.grid(True)

            self.canvas1.draw()
            self.canvas2.draw()

            max_error = np.max(np.abs(error))
            print(f"Eroarea maximă absolută: {max_error}")

        except Exception as e:
            messagebox.showerror("Error", f"A apărut o eroare: {str(e)}")
