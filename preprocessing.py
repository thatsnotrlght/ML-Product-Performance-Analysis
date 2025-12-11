import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = None
        self.clean_data = None
        self.log = [] 

    def load_data(self):
        try:
            self.raw_data = pd.read_csv(self.filepath)
            self.clean_data = self.raw_data.copy()
            self.log.append(f"Successfully loaded {len(self.raw_data)} records.")
            return True
        except FileNotFoundError:
            self.log.append("Error: File not found.")
            return False

    def handle_missing_values(self):
        """
        Analyzes and fills missing values.
        """
        if self.clean_data is None: return

        missing_cols = self.clean_data.columns[self.clean_data.isnull().any()].tolist()
        self.log.append(f"\n[Missing Values Analysis]\nFound missing data in: {missing_cols}")

        for col in missing_cols:
            if self.clean_data[col].dtype in ['int64', 'float64']:
                # Median
                median_val = self.clean_data[col].median()
                self.clean_data[col] = self.clean_data[col].fillna(median_val)
                self.log.append(f" - Imputed '{col}' with median: {median_val:.2f}")
            else:
                # Mode
                mode_val = self.clean_data[col].mode()[0]
                self.clean_data[col] = self.clean_data[col].fillna(mode_val)
                self.log.append(f" - Imputed '{col}' with mode: {mode_val}")

    def handle_outliers(self):
        """
        Detects and caps outliers using the IQR method manually.
        """
        if self.clean_data is None: return

        numerical_cols = ['price', 'cost', 'units_sold', 'promotion_frequency', 'profit']
        self.log.append("\n[Outlier Treatment]")

        for col in numerical_cols:
            #calculation of Quartiles
            Q1 = self.clean_data[col].quantile(0.25)
            Q3 = self.clean_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = self.clean_data[(self.clean_data[col] < lower_bound) | (self.clean_data[col] > upper_bound)]
            
            if not outliers.empty:
                self.clean_data[col] = np.where(self.clean_data[col] < lower_bound, lower_bound, self.clean_data[col])
                self.clean_data[col] = np.where(self.clean_data[col] > upper_bound, upper_bound, self.clean_data[col])
                self.log.append(f" - Capped {len(outliers)} outliers in '{col}' to range ({lower_bound:.2f}, {upper_bound:.2f})")

    def normalize_features(self):
        """
        Manually normalizes numerical features using Min-Max (0-1) formula.
        Formula: (Value - Min) / (Max - Min)
        """
        if self.clean_data is None: return

        features_to_scale = ['price', 'cost', 'units_sold', 'promotion_frequency', 'shelf_level']
        self.log.append("\n[Feature Normalization]")
        self.log.append(" - Applied Min-Max Normalization (0-1 scaling) manually.")

        for col in features_to_scale:
            min_val = self.clean_data[col].min()
            max_val = self.clean_data[col].max()
            
            if max_val != min_val:
                self.clean_data[col] = (self.clean_data[col] - min_val) / (max_val - min_val)
            else:
                self.clean_data[col] = 0.0 # If all values are the same, they become 0
            
        self.log.append(" - Reason: Essential for K-means distance calculations.")

    def run_pipeline(self):
        if self.load_data():
            self.handle_missing_values()
            self.handle_outliers()
            self.normalize_features()
            return self.clean_data, "\n".join(self.log)
        return None, "Failed to load data."