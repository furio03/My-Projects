import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, pairwise_distances

def auto_kmeans(df: pd.DataFrame, min_k=2, max_k=10)-> pd.DataFrame:
    """
    Performs K-Means clustering, automatically finding the optimal 'k' 
    using the Silhouette Score.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        min_k (int): Minimum number of clusters to test.
        max_k (int): Maximum number of clusters to test.
        
    Returns:
        pd.DataFrame: A dataframe containing the mean features for each group (profile).
        pd.DataFrame: The original dataframe with an additional 'Cluster' column.
    """
    if len(df)>6000 :
        df = df.sample(6000, random_state=42)
    
    # Ensure max_k doesn't exceed n_samples - 1 (required for silhouette_score)
    n_samples = len(df)
    max_k = min(max_k, n_samples - 1)
    
    # Ensure min_k is valid
    if min_k >= n_samples:
        min_k = max(2, n_samples - 1)

    # Not enough samples for clustering
    if n_samples < 3:
        return pd.DataFrame()

    #  Scaling: Crucial for K-Means (prevents variables with large scales from dominating)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    #  Find the optimal K
    best_k = min_k
    best_score = -1
    best_model = None
    
    for k in range(min_k, max_k + 1):
        # n_init='auto' suppresses future warnings, random_state ensures reproducibility
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(df_scaled)
        
        # Calculate Silhouette Score
        score = silhouette_score(df_scaled, cluster_labels)
        
        # If this score is better than the previous best, save this model
        if score > best_score:
            best_score = score
            best_k = k
            best_model = kmeans
            
    #  Final Application on the original DataFrame
    # Create a copy to avoid modifying the original dataframe outside the function
    df_result = df.copy()
    
    # Assign the labels from the best model
    df_result['Cluster'] = best_model.labels_
    
    #  Cluster Profiling (Mean of features)
    # Group by cluster and calculate the mean of numeric variables
    group_profile = df_result.groupby('Cluster').mean(numeric_only=True)
    
    # Add a column indicating the number of elements in each group
    group_profile['Count'] = df_result['Cluster'].value_counts()
    
    return group_profile

def _kmedoids_fit(X, k, metric='manhattan', max_iter=300):
    """
    K-Medoids implementation (PAM algorithm - Partitioning Around Medoids).
    
    Args:
        X: Normalized data array (n_samples, n_features)
        k: Number of clusters
        metric: Distance metric ('manhattan', 'euclidean', etc.)
        max_iter: Maximum number of iterations
        
    Returns:
        medoid_indices: Indices of medoids in original dataset
        labels: Cluster assignment for each point
    """
    n_samples = X.shape[0]
    
    # Initialization: select k random points as initial medoids
    np.random.seed(42)
    medoid_indices = np.random.choice(n_samples, k, replace=False)
    
    # Compute distance matrix (more efficient)
    distances = pairwise_distances(X, metric=metric)
    
    for iteration in range(max_iter):
        # Assign each point to nearest medoid
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            labels[i] = np.argmin(distances[i, medoid_indices])
        
        # Find best medoid for each cluster
        new_medoid_indices = []
        
        for cluster_id in range(k):
            cluster_points = np.where(labels == cluster_id)[0]
            
            if len(cluster_points) == 0:
                # If cluster is empty, select random point
                new_medoid_indices.append(np.random.choice(n_samples))
            else:
                # Calculate sum of distances for each point in cluster
                min_total_distance = float('inf')
                best_medoid = cluster_points[0]
                
                for point_idx in cluster_points:
                    total_distance = np.sum(distances[point_idx, cluster_points])
                    
                    if total_distance < min_total_distance:
                        min_total_distance = total_distance
                        best_medoid = point_idx
                
                new_medoid_indices.append(best_medoid)
        
        # If medoids don't change, convergence is reached
        if np.array_equal(np.sort(medoid_indices), np.sort(new_medoid_indices)):
            break
        
        medoid_indices = np.array(new_medoid_indices)
    
    # Final assignment
    labels = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        labels[i] = np.argmin(distances[i, medoid_indices])
    
    return medoid_indices, labels

def auto_kmedoids(df: pd.DataFrame, min_k=2, max_k=10, 
                  metric='manhattan', max_iter=300) -> pd.DataFrame:
    """
    K-Medoids clustering for CATEGORICAL/ORDINAL data with automatic encoding.
    
    📍 IDEAL FOR ORDINAL/CATEGORICAL DATA:
    - Automatic encoding: categories → ordinal numbers
    - Manhattan distance: preserves natural order
    - Real medoids: uses dataset points (not artificial averages)
    - Robust to outliers
    
    Args:
        df (pd.DataFrame): Input dataframe with categorical/ordinal variables
        min_k (int): Minimum number of clusters to test (default 2)
        max_k (int): Maximum number of clusters to test (default 10)
        metric (str): Distance metric ('manhattan', 'euclidean', 'cosine')
                     'manhattan' is OPTIMAL for ordinal/categorical data
        max_iter (int): Maximum iterations for convergence (default 300)
    
    Returns:
        pd.DataFrame: Medoid profiles (original categories) with 'Count' and 'Mode' columns
        
    Workflow:
        1. Encoding: Categories → Numbers (preserving order if ordinal)
        2. Normalization: Min-Max [0,1]
        3. K-Medoids: Clustering with Manhattan distance
        4. Decoding: Numbers → Original categories in profiles
    
    Example:
        >>> df = pd.DataFrame({
        ...     'Satisfaction': ['Low', 'Medium', 'High', 'Low', 'High'],
        ...     'Rating': [1, 2, 3, 4, 5]
        ... })
        >>> profiles = auto_kmedoids(df, min_k=2, max_k=3)
    """
    
    # Remove NaN values
    df_work = df.copy().dropna()
    
    if len(df_work) < 2:
        raise ValueError("Insufficient non-missing data for clustering")
    
    # Sampling (like auto_kmeans)
    if len(df_work) > 6000:
        df_work = df_work.sample(6000, random_state=42)
    
    n_samples = len(df_work)
    
    # Ensure max_k doesn't exceed n_samples - 1
    max_k = min(max_k, n_samples - 1)
    
    # Ensure min_k is valid
    if min_k >= n_samples:
        min_k = max(2, n_samples - 1)
    
    # ========================================================================
    # ENCODING: Categories → Numbers (preserving order)
    # ========================================================================
    df_encoded = df_work.copy()
    encoding_maps = {}  # To decode medoids at the end
    
    for col in df_work.columns:
        if df_work[col].dtype == 'object' or df_work[col].dtype.name == 'category':
            # Categorical variable: Label Encoding
            unique_vals = df_work[col].unique()
            
            # If ordinal (e.g., 'Low', 'Medium', 'High'), sort alphabetically
            # User can pre-sort categories if custom order is needed
            unique_vals_sorted = sorted(unique_vals)
            
            encoding_map = {val: idx for idx, val in enumerate(unique_vals_sorted)}
            encoding_maps[col] = encoding_map
            
            # Apply encoding
            df_encoded[col] = df_work[col].map(encoding_map)
        else:
            # Variable already numeric: keep as is
            encoding_maps[col] = None
    
    # ========================================================================
    # NORMALIZATION: Min-Max [0,1]
    # ========================================================================
    df_normalized = df_encoded.copy()
    normalization_params = {}
    
    for col in df_encoded.columns:
        min_val = df_encoded[col].min()
        max_val = df_encoded[col].max()
        range_val = max_val - min_val
        
        normalization_params[col] = {'min': min_val, 'max': max_val}
        
        if range_val > 0:
            df_normalized[col] = (df_encoded[col] - min_val) / range_val
        else:
            df_normalized[col] = 0
    
    X = df_normalized.values
    
    # ========================================================================
    # K-MEDOIDS: Find optimal K with Silhouette Score
    # ========================================================================
    best_k = min_k
    best_score = -1
    best_medoid_indices = None
    best_labels = None
    
    for k in range(min_k, max_k + 1):
        # K-Medoids algorithm
        medoid_indices, labels = _kmedoids_fit(X, k, metric=metric, max_iter=max_iter)
        
        # Calculate Silhouette Score
        score = silhouette_score(X, labels, metric=metric)
        
        if score > best_score:
            best_score = score
            best_k = k
            best_medoid_indices = medoid_indices
            best_labels = labels
    
    # ========================================================================
    # FINAL APPLICATION: Assign clusters
    # ========================================================================
    df_result = df_work.copy()
    df_result['Cluster'] = best_labels
    
    # ========================================================================
    # PROFILES: Medoids in ORIGINAL CATEGORIES (decoding)
    # ========================================================================
    # Extract medoids from ORIGINAL dataset (not encoded/normalized)
    medoid_profiles = df_work.iloc[best_medoid_indices].copy()
    medoid_profiles.index = range(best_k)
    
    # Add Count and Mode statistics for each cluster
    cluster_counts = df_result['Cluster'].value_counts().sort_index()
    medoid_profiles['Count'] = cluster_counts
    
    # Calculate mode (most frequent value) for each column in each cluster
    for col in df_work.columns:
        mode_values = []
        for cluster_id in range(best_k):
            cluster_data = df_result[df_result['Cluster'] == cluster_id][col]
            # Get the mode (most frequent value)
            mode_val = cluster_data.mode()
            if len(mode_val) > 0:
                mode_values.append(mode_val.iloc[0])
            else:
                mode_values.append(None)
        medoid_profiles[f'{col}_Mode'] = mode_values
    
    return medoid_profiles

