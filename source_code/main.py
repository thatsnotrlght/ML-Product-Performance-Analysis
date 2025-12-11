import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

from preprocessing import DataPreprocessor
from kmeans import KMeansManual, run_elbow_method, get_cluster_stats
from regression import RegressionModeler

class MachineLearningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Product Sales Analysis Dashboard")
        self.root.geometry("1200x800")

        self.status_var = tk.StringVar()
        self.status_var.set("Loading data...")
        
        # Initialize Preprocessor
        self.preprocessor = DataPreprocessor('data/product_sales.csv')
        self.clean_data, self.log_text = self.preprocessor.run_pipeline()
        
        # copy of raw data for display purposes (before normalization)
        self.raw_data = self.preprocessor.raw_data

        # Tab creation
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        self.create_overview_tab()
        self.create_clustering_tab()
        self.create_regression_tab()

    def create_overview_tab(self):
        """Tab 1: Data Overview"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="1. Data Overview")

        metrics_frame = ttk.LabelFrame(tab, text="Dataset Health")
        metrics_frame.pack(fill='x', padx=15, pady=10)

        if self.raw_data is not None:
            # Metric 1: Total Records
            lbl_records = tk.Label(metrics_frame, text="Total Records", font=("Arial", 10, "bold"), fg="#555")
            lbl_records.grid(row=0, column=0, padx=20, pady=5)
            val_records = tk.Label(metrics_frame, text=f"{len(self.raw_data)}", font=("Arial", 18, "bold"), fg="blue")
            val_records.grid(row=1, column=0, padx=20, pady=(0, 10))

            # Metric 2: Feature Count
            lbl_features = tk.Label(metrics_frame, text="Features", font=("Arial", 10, "bold"), fg="#555")
            lbl_features.grid(row=0, column=1, padx=20, pady=5)
            val_features = tk.Label(metrics_frame, text=f"{len(self.raw_data.columns)}", font=("Arial", 18, "bold"), fg="green")
            val_features.grid(row=1, column=1, padx=20, pady=(0, 10))

            # Metric 3: Missing Values Status
            has_missing = self.raw_data.isnull().sum().sum() > 0
            lbl_missing = tk.Label(metrics_frame, text="Missing Data", font=("Arial", 10, "bold"), fg="#555")
            lbl_missing.grid(row=0, column=2, padx=20, pady=5)
            
            status_text = "Detected & Fixed" if has_missing else "None"
            status_color = "#e67e22" if has_missing else "gray" # Orange if fixed, Gray if clean
            val_missing = tk.Label(metrics_frame, text=status_text, font=("Arial", 18, "bold"), fg=status_color)
            val_missing.grid(row=1, column=2, padx=20, pady=(0, 10))

        stats_frame = ttk.LabelFrame(tab, text="Numerical Statistics Summary")
        stats_frame.pack(fill='both', expand=True, padx=15, pady=10)

        # Treeview for Statistics
        if self.raw_data is not None:
            desc_df = self.raw_data.describe().T.reset_index()
            desc_df.rename(columns={'index': 'Feature'}, inplace=True)
            
            # Define columns
            cols = list(desc_df.columns)
            tree = ttk.Treeview(stats_frame, columns=cols, show='headings', height=8)
            
            # Create Scrollbar
            vsb = ttk.Scrollbar(stats_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=vsb.set)
            
            # Setup Headings
            for col in cols:
                tree.heading(col, text=col.capitalize())
                width = 150 if col == 'Feature' else 80
                tree.column(col, width=width, anchor='center')

            # Insert Data
            for _, row in desc_df.iterrows():
                formatted_vals = [row['Feature']] + [f"{x:.2f}" for x in row[1:]]
                tree.insert("", "end", values=formatted_vals)

            tree.pack(side='left', fill='both', expand=True, padx=5, pady=5)
            vsb.pack(side='right', fill='y', pady=5)

        log_frame = ttk.LabelFrame(tab, text="Preprocessing Actions Log")
        log_frame.pack(fill='x', padx=15, pady=10)
        
        txt_log = scrolledtext.ScrolledText(log_frame, height=6, font=("Consolas", 9), bg="#f4f4f4")
        txt_log.insert(tk.END, self.log_text)
        txt_log.config(state='disabled') 
        txt_log.pack(fill='both', padx=5, pady=5)

    def create_clustering_tab(self):
        """Tab 2: Clustering Results [cite: 208]"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="2. Clustering Analysis")

        # Split: Left (Plots), Right (Stats & Interpretation)
        paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill='both', expand=True)

        left_frame = ttk.Frame(paned)
        right_frame = ttk.Frame(paned, width=400)
        paned.add(left_frame, weight=3)
        paned.add(right_frame, weight=1)

        # --- Plots (Matplotlib) ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        
        # 1. Elbow Curve
        # numerical features for clustering
        features = ['price', 'units_sold', 'promotion_frequency', 'shelf_level', 'cost']
        X = self.clean_data[features].values
        
        elbow_results = run_elbow_method(X, max_k=8)
        ks = list(elbow_results.keys())
        wcss = list(elbow_results.values())
        
        ax1.plot(ks, wcss, 'bo-', markersize=8)
        ax1.set_title('Elbow Method for Optimal K')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('WCSS')
        ax1.grid(True)

        # 2. Cluster Visualization (Scatter Plot)
        # Run K-Means with optimal K (assuming k=3)
        k_optimal = 3 
        kmeans = KMeansManual(k=k_optimal)
        labels, centroids = kmeans.fit(X)
        
        # We plot 'price' (index 0) vs 'units_sold' (index 1)
        scatter = ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
        ax2.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
        ax2.set_title(f'Cluster Visualization (K={k_optimal})')
        ax2.set_xlabel('Price (Normalized)')
        ax2.set_ylabel('Units Sold (Normalized)')
        ax2.legend()

        canvas = FigureCanvasTkAgg(fig, master=left_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # --- Statistics & Interpretation ---
        
        lbl_title = tk.Label(right_frame, text=f"Cluster Statistics (K={k_optimal})", font=("Arial", 12, "bold"))
        lbl_title.pack(pady=10)

        # Calculate readable stats using original data
        stats_df = get_cluster_stats(self.raw_data, labels, k_optimal)
        
        # Create Treeview for table
        cols = list(stats_df.columns)
        tree = ttk.Treeview(right_frame, columns=cols, show='headings', height=5)
        
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=90)  # Adjust width

        for index, row in stats_df.iterrows():
            tree.insert("", "end", values=list(row))
        
        tree.pack(fill='x', padx=5)

        # Interpretation Text
        interp_frame = ttk.LabelFrame(right_frame, text="Interpretation")
        interp_frame.pack(fill='both', expand=True, padx=5, pady=10)
        
        interp_text = (
            "INTERPRETATION GUIDANCE:\n\n"
            "Cluster 0: Look at the Avg Price and Units Sold.\n"
            "If Price is low and Units are high, these are likely 'Budget Best-Sellers'.\n\n"
            "Cluster 1: Look at Profit.\n"
            "High profit items might be 'Premium' goods.\n\n"
            "Business Insight:\n"
            "- Ensure stock levels for high-volume clusters.\n"
            "- Create promotions for low-volume, high-margin items."
        )
        lbl_interp = tk.Label(interp_frame, text=interp_text, justify="left", wraplength=300)
        lbl_interp.pack(anchor='nw', padx=5, pady=5)

    def create_regression_tab(self):
        """Tab 3: Regression Results [cite: 213]"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="3. Regression Analysis")

        # Initialize Regression Modeler
        reg_modeler = RegressionModeler(self.clean_data)
        reg_modeler.prepare_data()
        
        # Run Models
        res_linear = reg_modeler.run_linear_regression()
        res_poly = reg_modeler.run_polynomial_regression(degree=2)

        
        # 1. Model Comparison Table
        table_frame = ttk.LabelFrame(tab, text="Model Performance Comparison")
        table_frame.pack(fill='x', padx=10, pady=5)
        
        cols = ["Model", "MSE (Lower is better)", "MAE (Lower is better)"]
        tree = ttk.Treeview(table_frame, columns=cols, show='headings', height=2)
        
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=200)

        # Insert Data
        tree.insert("", "end", values=(res_linear['Model'], f"{res_linear['MSE']:.4f}", f"{res_linear['MAE']:.4f}"))
        tree.insert("", "end", values=(res_poly['Model'], f"{res_poly['MSE']:.4f}", f"{res_poly['MAE']:.4f}"))
        tree.pack(padx=10, pady=10)

        # 2. Actual vs Predicted Plot
        plot_frame = ttk.Frame(tab)
        plot_frame.pack(fill='both', expand=True, padx=10, pady=10)

        fig, ax = plt.subplots(figsize=(6, 4))
        
        y_true, y_pred = reg_modeler.get_results_for_plot('linear')
        
        ax.scatter(y_true, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')
        
        # Diagonal Line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_title("Regression Performance: Actual vs Predicted Profit")
        ax.set_xlabel("Actual Profit (Normalized)")
        ax.set_ylabel("Predicted Profit (Normalized)")
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = MachineLearningApp(root)
    root.mainloop()