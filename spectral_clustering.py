"""
Work with Spectral clustering.
Do not use global variables!
"""

from numpy.linalg import norm
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from scipy.special import comb
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
    # Check centroid shape
    if centroids.shape[1] != data.shape[1]:
        raise ValueError("Centroid and data dimension mismatch")
    
    for k in range(centroids.shape[0]):
        cluster_data = data[labels == k]
        if cluster_data.size == 0:
            continue  # Skip empty clusters
        distances = np.linalg.norm(cluster_data - centroids[k], axis=1)
        sse += np.sum(distances**2)
    return sse



def compute_ari(true_labels, computed_labels):
    # Create a contingency table
    contingency_matrix = np.zeros((int(true_labels.max()) + 1, int(computed_labels.max()) + 1))
    for true_label, computed_label in zip(true_labels, computed_labels):
        contingency_matrix[int(true_label), int(computed_label)] += 1

    # Sum over rows & columns
    sum_over_rows = np.sum(contingency_matrix, axis=1)
    sum_over_cols = np.sum(contingency_matrix, axis=0)
    total = sum(sum_over_rows)

    # Sum over pairs
    sum_over_pairs = np.sum(contingency_matrix * (contingency_matrix - 1)) / 2
    sum_over_rows_pairs = np.sum(sum_over_rows * (sum_over_rows - 1)) / 2
    sum_over_cols_pairs = np.sum(sum_over_cols * (sum_over_cols - 1)) / 2

    # Calculate expected index
    expected_index = sum_over_rows_pairs * sum_over_cols_pairs / (total * (total - 1) / 2)
    
    # Calculate max index
    max_index = (sum_over_rows_pairs + sum_over_cols_pairs) / 2

    # Adjusted Rand Index
    ari = (sum_over_pairs - expected_index) / (max_index - expected_index)

    return ari




def k_means(data, k, max_iters=100):
    # Initialize centroids to random points
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    for iteration in range(max_iters):
        # Assign points to the nearest centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        nearest_centroids = np.argmin(distances, axis=0)

        # Recompute centroids
        new_centroids = []
        for i in range(k):
            cluster_points = data[nearest_centroids == i]
            if cluster_points.size == 0:
                # Reinitialize the centroid randomly if no points are assigned to it
                new_centroid = data[np.random.randint(len(data))].copy()  # Ensure it's a copy of the point
                new_centroids.append(new_centroid)
            else:
                # Otherwise, compute the mean of all points assigned to this centroid
                new_centroid = cluster_points.mean(axis=0)
                new_centroids.append(new_centroid)

        new_centroids = np.array(new_centroids)

        # Check for convergence
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    return nearest_centroids, centroids





def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[
    NDArray[np.int32] | None, float | None, float | None, NDArray[np.floating] | None
]:
    sigma = params_dict.get('sigma', 1.0)
    k = params_dict.get('k', 5)
    n_clusters = 5

    affinity_matrix = compute_affinity_matrix(data, sigma)
    affinity_matrix = sparsify_affinity_matrix(affinity_matrix, k)
    laplacian = compute_laplacian(affinity_matrix)
    eigenvalues, eigenvectors = eigsh(laplacian, k=n_clusters, which='SM', tol=1e-3)

    computed_labels, centroids = k_means(eigenvectors, n_clusters)

    
    SSE = compute_sse(eigenvectors, computed_labels, centroids)  
    ARI = compute_ari(labels, computed_labels)

    return computed_labels, SSE, ARI, eigenvalues


def spectral_clustering():
    """
    Performs DENCLUE clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    data = np.load('question1_cluster_data.npy')
    labels = np.load('question1_cluster_labels.npy')

    answers = {}

    # Return your `spectral` function
    answers["spectral_function"] = spectral

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    sigmas = np.linspace(0.1, 10, 20)  # Choose 20 values for sigma within the range
    first_group_data = data[:1000]
    first_group_labels = labels[:1000]

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}

    best_ari = -np.inf
    best_sse = np.inf
    best_sigma = None
    ari_values = []
    sse_values = []

    for sigma in sigmas:
        params_dict = {'sigma': sigma, 'k': 5}
        computed_labels, SSE, ARI, eigenvalues = spectral(first_group_data, first_group_labels, params_dict)
        groups[sigma] = {"ARI": ARI, "SSE": SSE}
        ari_values.append(ARI)
        sse_values.append(SSE)

        if ARI > best_ari:
            best_ari = ARI
            best_ari_sigma = sigma

        if SSE < best_sse:
            best_sse = SSE
            best_sse_sigma = sigma

    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[sigmas[0]]["SSE"]

    # Identify the cluster with the lowest value of ARI. This implies
    # that you set the cluster number to 5 when applying the spectral
    # algorithm.

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Plotting scatter plots for ARI and SSE
    plt.figure(figsize=(12, 6))
    plot_ARI = plt.scatter(sigmas, ari_values, c='green')
    plt.xlabel('Sigma')
    plt.ylabel('ARI')
    plt.title('ARI by Sigma')
    plt.grid(True)

    plt.figure(figsize=(12, 6))
    plot_SSE = plt.scatter(sigmas, sse_values, c='blue')
    plt.xlabel('Sigma')
    plt.ylabel('SSE')
    plt.title('SSE by Sigma')
    plt.grid(True)

    

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.
    _, _, _, eigenvalues = spectral(first_group_data, first_group_labels, {'sigma': best_ari_sigma, 'k': 5})
    plt.figure(figsize=(12, 6))
    plot_eig = plt.plot(range(len(eigenvalues)), sorted(eigenvalues))
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Sorted Eigenvalues')
    plt.grid(True)
    answers["eigenvalue plot"] = plot_eig


    # Applying the best sigma to other data slices
    aris = [best_ari]  # Start with best ARI from the first group
    sses = [best_sse]  # Start with best SSE from the first group
    for i in range(1, 5):
        slice_data = data[i * 1000: (i + 1) * 1000]
        slice_labels = labels[i * 1000: (i + 1) * 1000]
        computed_labels, SSE, ARI, _ = spectral(slice_data, slice_labels, {'sigma': best_ari_sigma, 'k': 5})
        aris.append(ARI)
        sses.append(SSE)

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    # A single float
    answers["mean_ARIs"] = np.mean(aris)

    # A single float
    answers["std_ARIs"] = np.std(aris)

    # A single float
    answers["mean_SSEs"] = np.mean(sses)

    # A single float
    answers["std_SSEs"] = np.std(sses)

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
