import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error

class RegressionModeler:
    def __init__(self, data):
        """
        Initialize with the full dataset
        """
        self.data = data
        self.model_linear = None
        self.model_poly = None
        self.poly_features = None
        self.X_test = None
        self.y_test = None
        self.predictions_linear = None
        self.predictions_poly = None
        
        # We predict 'profit' using the other numerical features
        self.feature_cols = ['price', 'cost', 'units_sold', 'promotion_frequency', 'shelf_level']
        self.target_col = 'profit'

    def prepare_data(self):
        """
        Splits data into Training (70%) and Testing (30%) sets.
        """
        # Select only the numeric columns needed
        X = self.data[self.feature_cols]
        y = self.data[self.target_col]

        # random_state=42 ensures the split is the same every time we run it
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        return True

    def run_linear_regression(self):
        """
        Model 1: Standard Linear Regression
        """
        self.model_linear = LinearRegression()
        self.model_linear.fit(self.X_train, self.y_train)
        
        self.predictions_linear = self.model_linear.predict(self.X_test)
        
        return self.evaluate_model(self.y_test, self.predictions_linear, "Linear Regression")

    def run_polynomial_regression(self, degree=2):
        """
        Model 2: Polynomial Regression
        Captures non-linear relationships (if profit grows exponentially with sales).
        """
        # Transform features to polynomial terms (x^2, xy, etc.
        self.poly = PolynomialFeatures(degree=degree)
        X_train_poly = self.poly.fit_transform(self.X_train)
        X_test_poly = self.poly.transform(self.X_test)

        # Train Linear Regression on the transformed features
        self.model_poly = LinearRegression()
        self.model_poly.fit(X_train_poly, self.y_train)
        
        # Make predictions
        self.predictions_poly = self.model_poly.predict(X_test_poly)
        
        return self.evaluate_model(self.y_test, self.predictions_poly, f"Polynomial Regression (Deg {degree})")

    def evaluate_model(self, y_true, y_pred, model_name):
        """
        Calculates MSE and MAE for evaluation. [cite: 181-183]
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        return {
            "Model": model_name,
            "MSE": mse, # Average squared error (punishes large errors more)
            "MAE": mae  # Average absolute error (easier to interpret in $)
        }

    def get_results_for_plot(self, model_type='linear'):
        """
        Returns data needed for the 'Actual vs Predicted' scatter plot
        """
        if model_type == 'linear':
            return self.y_test, self.predictions_linear
        else:
            return self.y_test, self.predictions_poly