import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
def spectral_clustering_mean_shift(weight_matrix):
    """
    Perform spectral clustering on a given weight matrix and use Mean Shift for clustering.

    Parameters:
        weight_matrix (np.ndarray): N x N symmetric weight matrix.

    Returns:
        labels (np.ndarray): Cluster labels for each data point.
        n_clusters (int): Number of clusters found by Mean Shift.
    """
    N = weight_matrix.shape[0]

    # Ensure the matrix is symmetric
    weight_matrix = (weight_matrix + weight_matrix.T) / 2

    # Step 1: Construct the Laplacian matrix
    degree_matrix = np.diag(np.sum(weight_matrix, axis=1))  # Degree matrix
    laplacian_matrix = degree_matrix - weight_matrix  # Unnormalized Laplacian

    # Step 2: Compute the first few eigenvectors of the Laplacian
    # Use a reasonable number of eigenvectors (e.g., 10 or N, whichever is smaller)
    n_eigenvectors = min(10, N)
    eigenvalues, eigenvectors = eigh(laplacian_matrix, subset_by_index=[0, n_eigenvectors - 1])

    # Step 3: Use the eigenvectors as the new feature space
    embedding = eigenvectors

    # Step 4: Perform Mean Shift clustering on the embedding
    mean_shift = MeanShift(bandwidth=None, bin_seeding=True)  # Let Mean Shift automatically determine bandwidth
    mean_shift.fit(embedding)
    labels = mean_shift.labels_
    n_clusters = len(np.unique(labels))  # Number of clusters found

    return labels, n_clusters

def spectral_clustering(weight_matrix, n_clusters):
    """
    Perform spectral clustering on a given weight matrix.

    Parameters:
        weight_matrix (np.ndarray): N x N symmetric weight matrix.
        n_clusters (int): Number of clusters to form.

    Returns:
        labels (np.ndarray): Cluster labels for each data point.
    """
    N = weight_matrix.shape[0]

    # Ensure the matrix is symmetric
    weight_matrix = (weight_matrix + weight_matrix.T) / 2

    # Step 1: Construct the Laplacian matrix
    degree_matrix = np.diag(np.sum(weight_matrix, axis=1))  # Degree matrix
    laplacian_matrix = degree_matrix - weight_matrix  # Unnormalized Laplacian

    # Step 2: Compute the first `n_clusters` eigenvectors of the Laplacian
    eigenvalues, eigenvectors = eigh(laplacian_matrix, subset_by_index=[0, n_clusters - 1])

    # Step 3: Use the eigenvectors as the new feature space
    embedding = eigenvectors

    # Step 4: Perform k-means clustering on the embedding
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embedding)

    return labels

# Example usage
if __name__ == "__main__":
    # Example weight matrix (N x N)
    N = 5
    weight_matrix = np.eye(N)  # Diagonal elements are 1
    off_diagonal_sum = 1  # Sum of off-diagonal elements is 1
    weight_matrix[0, 1] = off_diagonal_sum
    weight_matrix[1, 0] = off_diagonal_sum  # Make the matrix symmetric

    print("Weight Matrix:")
    print(weight_matrix)

    # Perform spectral clustering
    n_clusters = 2  # Number of clusters
    labels = spectral_clustering(weight_matrix, n_clusters)

    print("Cluster Labels:")
    print(labels)