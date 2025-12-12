### Product Sales Analysis Dashboard

#### Author Information

- **Name**: Cristian Aldana, Samuel Artiste
- **Student ID**: 6426411, 6538723
- **Course**: CAI 4002 - Artificial Intelligence
- **Semester**: Fall 2025



#### System Overview

Our project was made to analyze supermarket product sales containing **200 records** using machine learning techniques discussed in class. We built the system using **Python** as the main development language and integrated a **Tkinter GUI dashboard**.

The system:

- Implements a **custom K-Means clustering algorithm** to segment products based on features like **price** and **units sold**, allowing business-level labels such as “Budget”, “Mid-range”, and “Premium”.
- Applies **regression models** to **predict monthly profit**, comparing the performance of:
  - **Linear Regression**
  - **Polynomial Regression (Degree 2)**

The emphasis of the project is on understanding the **mechanics** of the algorithms by:

- Implementing **K-Means from scratch**
- Using **scikit-learn** for Linear and Polynomial Regression and comparing their performance



#### Technical Stack

- **Language**: Python 3.x
- **Key Libraries**:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
- **UI Framework**: Tkinter



#### Installation

##### Prerequisites
- Python 3.8+ installed

##### Setup

No complex setup is required beyond cloning and installing dependencies.

Clone or extract the project:

- Repository:
    https://github.com/thatsnotrlght/ML-Product-Performance-Analysis.git

Install dependencies:

- From your terminal:

    pip install pandas
    pip install numpy
    pip install matplotlib
    pip install -U scikit-learn

Run application:

- From the project root, run:

    python main.py



#### Usage

The application is divided into **three main tabs**:

##### 1. Data Overview

- Automatically loads `product_sales.csv` on launch.
- Displays dataset health metrics:
  - Total number of records
  - Number of features
  - Presence or absence of missing values
- Shows a **log of preprocessing actions**, for example:
  - “Imputed `price` with median”
  - “Scaled `units_sold` with Min-Max Scaling”
- Provides a **scrollable table** of numerical statistics (mean, std, min, max, quartiles) for all numerical features.

##### 2. Clustering Analysis

- Implements **K-Means clustering** on selected features such as:
  - Price
  - Units Sold
  - Profit (if included)
- Elbow Method:
  - Displays the **Within-Cluster Sum of Squares (WCSS)** curve to help choose the optimal number of clusters `k`.
- Cluster Visualization:
  - Plots a **scatter diagram** (e.g., Price vs. Units Sold) with products colored according to their cluster.
- Cluster Statistics Table:
  - For each cluster, shows:
    - Average price
    - Average units sold
    - Average profit
  - Useful for interpreting clusters as **“Budget”**, **“Mid-range”**, or **“Premium”** segments.

##### 3. Regression Analysis

- Trains two models to **predict product profit**:
  - **Linear Regression** – baseline model.
  - **Polynomial Regression (Degree 2)** – captures non-linear relationships (e.g., price and units interactions).
- Comparison Table:
  - Displays performance metrics:
    - MSE (Mean Squared Error)
    - MAE (Mean Absolute Error)
  - For both Linear and Polynomial models.
- Prediction Plot:
  - Visualizes **Actual vs. Predicted** profit to see how well the model fits the data.



#### Algorithm Implementation

##### K-Means (Manual Implementation)

The **K-Means clustering** algorithm is implemented **from scratch** in `kmeans.py` without using high-level clustering libraries.

Core ideas:

- **Initialization**:
  - Randomly selects `k` data points as the initial centroids.

- **Assignment Step**:
  - For each data point, computes the **Euclidean distance** to each centroid.
  - Uses **NumPy vectorization** for efficient distance computation.
  - Assigns each point to the nearest centroid.

- **Update Step**:
  - For each cluster, recomputes the centroid as the **mean** of all points assigned to that cluster.

- **Convergence**:
  - Repeats Assignment and Update steps until:
    - The movement of centroids is below a tolerance threshold (e.g., `1e-4`), or
    - A maximum number of iterations is reached.

This manual implementation allows fine-grained understanding of how K-Means works internally, including centroid updates and distance calculations.

##### Regression Analysis

The regression logic is handled in `regression.py` using **scikit-learn**:

- **Linear Regression**:
  - Uses features such as:
    - Price
    - Cost
    - Units Sold
  - Provides a **baseline** for predicting profit.

- **Polynomial Regression (Degree 2)**:
  - Uses `PolynomialFeatures` to generate:
    - Squared terms (e.g., `price^2`)
    - Interaction terms (e.g., `price × units_sold`)
  - Trains a regression model on this expanded feature set.
  - Better captures the **non-linear nature** of profit calculations.
  - Consistently provides **lower MSE and MAE** compared to the linear model on this dataset.



#### Performance Results

Tested on the `product_sales.csv` dataset (200 records):

- Polynomial Regression (Degree 2) generally:
  - Achieves **lower MSE** and **lower MAE** than Linear Regression.
  - Provides a closer fit between **actual** and **predicted** profits.

High-level comparison:

| Model                         | Relative Performance           |
|------------------------------|--------------------------------|
| Linear Regression            | Higher error; baseline model   |
| Polynomial Regression (deg2) | Lower error; better fit        |



#### Project Structure

    project-root/
    ├── source_code/
    │   └── main.py           # Entry point; Tkinter GUI logic
    │   └── kmeans.py         # Manual implementation of K-Means algorithm
    │   └── regression.py     # Linear & Polynomial regression logic
    │   └── preprocessing.py  # Data cleaning and normalization pipeline      
    ├── data/
    │   └── product_sales.csv # Dataset
    ├── README.md             # Project documentation
    └── REPORT.pdf            # Final analysis report



#### Data Preprocessing

Preprocessing logic is implemented in `preprocessing.py`. It handles common real-world data issues automatically.

Operations:

- **Missing Values**:
  - Numeric features:
    - Filled with **median** values.
  - Categorical features:
    - Filled with **mode** (most frequent) values.

- **Outliers**:
  - Detected via **IQR (Interquartile Range)** method.
  - Outliers are **capped (Winsorized)** to reduce their effect on:
    - K-Means clustering
    - Regression models

- **Normalization**:
  - Applies **Min-Max Scaling** to all numerical features, mapping them to the **[0, 1]** range.
  - This is critical for K-Means so that features with larger raw ranges (e.g., `units_sold` in [0, 1000]) do not overpower features like `price` in [0, 20].

Example effects:

- After preprocessing:
  - Number of missing numeric values → 0
  - All numeric features scaled between 0 and 1
  - Extreme outliers reduced in impact



#### Testing

Verified functionality:

- [✓] CSV loading and basic error handling
- [✓] Preprocessing steps (missing values, outlier handling, normalization)
- [✓] Manual K-Means clustering and convergence
- [✓] Regression models (Linear and Polynomial) and metric computation
- [✓] Tkinter GUI interaction across all tabs (Data Overview, Clustering, Regression)

Test cases:

| Component      | Test Input                        | Expected Outcome                                                          |
|----------------|-----------------------------------|----------------------------------------------------------------------------|
| Preprocessing  | Missing values in `price`         | Missing values are filled with the **median price**; no NaNs remain       |
| K-Means        | `k = 3`                           | Segmentation into “Budget”, “Mid-range”, and “Premium” product groups     |
| Regression     | Polynomial Regression (Degree 2)  | **MAE and MSE decrease** relative to the Linear Regression baseline       |
| Normalization  | Features with different scales    | All numeric features scaled to **[0, 1]**, improving K-Means performance  |



#### Known Limitations

- **Dataset Size**:
  - The dataset has **only 200 records**, which is relatively small.
  - This can increase the risk of **overfitting**, especially for the Polynomial Regression model.

- **Manual K-Means Performance**:
  - The K-Means implementation uses Python + NumPy loops and vectorization.
  - For very large datasets, it will be **slower** than optimized, C-based implementations such as `sklearn.cluster.KMeans`.



#### AI Tool Usage

During the implementation of this project, we used AI tools such as **Google Gemini** and **GitHub Copilot** for:

- **Debugging**:
  - Identifying and fixing vectorization issues in the manual K-Means distance calculation.
- **UI Prototyping**:
  - Generating boilerplate for the **Tkinter tabbed interface** (Data, Clustering, Regression tabs).
- **Concept Explanation**:
  - Clarifying the differences between **Standard Scaling** and **Min-Max Normalization** in the context of K-Means.
- **Documentation Support**:
  - Assisting in structuring both the **REPORT.pdf** and this **README**.

AI tools supported the development and debugging process, while we were responsible for understanding, integrating, and adapting the code and algorithms to fit the project requirements.



#### References

- Course lecture materials (CAI 4002 - Artificial Intelligence)
- Google Gemini AI
- GitHub Copilot
- Pandas Documentation
- NumPy Documentation
- Scikit-Learn Documentation
- Tkinter Documentation
- Matplotlib Documentation
