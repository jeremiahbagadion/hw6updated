"""
Work with Spectral clustering.
Do not use global variables!
"""

from numpy.linalg import norm
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def compute_affinity_matrix(data, sigma):
    """
    Computes the affinity matrix for the given data and sigma using the Gaussian (RBF) kernel.
    """
    # Calculate the squared Euclidean distances for every pair of points
    sq_dists = squareform(pdist(data, 'sqeuclidean'))

    # Compute the affinity matrix using the RBF kernel
    affinity_matrix = np.exp(-sq_dists / (sigma**2))

    return affinity_matrix

def sparsify_affinity_matrix(affinity_matrix, k):
    """
    Sparsifies the affinity matrix by keeping only the k-nearest neighbors for each point.
    """
    n = affinity_matrix.shape[0]
    for i in range(n):
        k_neighbors = np.argsort(affinity_matrix[i])[-(k+1):]
        for j in range(n):
            if j not in k_neighbors and i != j:
                affinity_matrix[i, j] = 0
    return affinity_matrix

def compute_laplacian(affinity_matrix):
    """
    Computes the unnormalized graph Laplacian of the affinity matrix.
    """
    degree_matrix = np.diag(affinity_matrix.sum(axis=1))
    laplacian = degree_matrix - affinity_matrix
    return laplacian

def compute_sse(data, labels, centroids):
    sse = 0.0
    for k in range(centroids.shape[0]):
        cluster_data = data[labels == k]
        distances = np.linalg.norm(cluster_data - centroids[k], axis=1)
        sse += np.sum(distances**2)
    return sse

def compute_ari(true_labels, computed_labels):
    # Create a contingency table
    contingency_matrix = np.zeros((true_labels.max() + 1, computed_labels.max() + 1))
    for true_label, computed_label in zip(true_labels, computed_labels):
        contingency_matrix[true_label, computed_label] += 1

    # Sum over rows & columns
    sum_over_rows = np.sum(contingency_matrix, axis=1)
    sum_over_cols = np.sum(contingency_matrix, axis=0)

    # Sum over pairs
    sum_over_pairs = np.sum(contingency_matrix * (contingency_matrix - 1)) / 2
    sum_over_rows_pairs = np.sum(sum_over_rows * (sum_over_rows - 1)) / 2
    sum_over_cols_pairs = np.sum(sum_over_cols * (sum_over_cols - 1)) / 2

    # Calculate ARI components
    total_pairs = np.sum(sum_over_rows * (sum_over_rows - 1)) / 2
    expected_index = sum_over_rows_pairs * sum_over_cols_pairs / total_pairs
    max_index = (sum_over_rows_pairs + sum_over_cols_pairs) / 2
    ari = (sum_over_pairs - expected_index) / (max_index - expected_index)

    return ari

def k_means(data, k, max_iters=100):
    # Initialize centroids to random points
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    for _ in range(max_iters):
        # Assign points to the nearest centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        nearest_centroids = np.argmin(distances, axis=0)
        # Recompute centroids
        new_centroids = np.array([data[nearest_centroids == i].mean(axis=0) for i in range(k)])
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return nearest_centroids, centroids

def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[
    NDArray[np.int32] | None, float | None, float | None, NDArray[np.floating] | None
]:
    """
    Implementation of the Spectral clustering  algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'k'. There could be others.
       params_dict['sigma']:  in the range [.1, 10]
       params_dict['k']: the number of clusters, set to five.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index
    - eigenvalues: eigenvalues of the Laplacian matrix
    """

    sigma = params_dict.get('sigma', 1.0)  
    k = params_dict.get('k', 5)            
    n_clusters = 5 

    # Step 1: Compute the affinity matrix
    affinity_matrix = compute_affinity_matrix(data, sigma)

    # Step 2: Sparsify the affinity matrix
    affinity_matrix = sparsify_affinity_matrix(affinity_matrix, k)

    # Step 3: Compute the Laplacian matrix
    laplacian = compute_laplacian(affinity_matrix)

    # Step 4: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigsh(laplacian, k=n_clusters, which='SM', tol=1e-3)

    # Step 5: Perform k-means clustering on the rows of the eigenvectors
    computed_labels, centroids = k_means(eigenvectors, n_clusters)

    # Step 6: Compute SSE
    SSE = compute_sse(data, computed_labels, centroids)

    # Step 7: Compute ARI
    ARI = compute_ari(labels, computed_labels)

    return computed_labels, SSE, ARI, eigenvalues


def spectral_clustering():
    """
    Performs DENCLUE clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    # Return your `spectral` function
    answers["spectral_function"] = spectral

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}

    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = {}

    # Identify the cluster with the lowest value of ARI. This implies
    # that you set the cluster number to 5 when applying the spectral
    # algorithm.

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    plot_ARI = plt.scatter([1,2,3], [4,5,6])
    plot_SSE = plt.scatter([1,2,3], [4,5,6])
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.
    plot_eig = plt.plot([1,2,3], [4,5,6])
    answers["eigenvalue plot"] = plot_eig

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    # A single float
    answers["mean_ARIs"] = 0.

    # A single float
    answers["std_ARIs"] = 0.

    # A single float
    answers["mean_SSEs"] = 0.

    # A single float
    answers["std_SSEs"] = 0.

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
