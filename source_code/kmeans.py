import numpy as np
import pandas as pd

class KMeansManual:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.wcss = 0  # Within-Cluster Sum of Squares (for Elbow Method)

    def fit(self, data):
        """
        Runs the K-means algorithm on the provided data
        """
        X = np.array(data)
        n_samples, n_features = X.shape

        # Randomly select 'k' data points from the dataset to start as centroids
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # calculate Euclidean distance from every point to every centroid.
            # self.centroids reshapes to (1, k, features)
            distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
            
            # Find the index of the minimum distance for each point (0 to k-1)
            new_labels = np.argmin(distances, axis=1)

            # Calculate the mean of all points assigned to each cluster
            new_centroids = np.array([
                X[new_labels == j].mean(axis=0) if np.any(new_labels == j) else self.centroids[j]
                for j in range(self.k)
            ])

            # If centroids haven't moved significantly we stop early.
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                self.labels = new_labels
                break
            
            self.centroids = new_centroids
            self.labels = new_labels

        # Calculate WCSS for the final clusters
        final_distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        min_dist_squared = np.min(final_distances, axis=1) ** 2
        self.wcss = np.sum(min_dist_squared)

        return self.labels, self.centroids

def run_elbow_method(data, max_k=8):
    """
    Runs K-means for k=2 to max_k and returns the WCSS for each
    """
    results = {}
    for k in range(2, max_k + 1):
        kmeans = KMeansManual(k=k)
        kmeans.fit(data)
        results[k] = kmeans.wcss
    return results

def get_cluster_stats(original_data, labels, k):
    """
    Calculates statistics for Step 2.3
    """
    # Create a temporary dataframe to group by cluster
    df_stats = original_data.copy()
    df_stats['Cluster'] = labels
    
    stats_list = []
    
    for i in range(k):
        cluster_data = df_stats[df_stats['Cluster'] == i]
        
        # Calculate the required averages
        stats = {
            'Cluster ID': i,
            'Count': len(cluster_data),
            'Avg Price': f"${cluster_data['price'].mean():.2f}",
            'Avg Units Sold': f"{cluster_data['units_sold'].mean():.0f}",
            'Avg Profit': f"${cluster_data['profit'].mean():.2f}",
            'Avg Promo Freq': f"{cluster_data['promotion_frequency'].mean():.1f}"
        }
        stats_list.append(stats)
        
    return pd.DataFrame(stats_list)