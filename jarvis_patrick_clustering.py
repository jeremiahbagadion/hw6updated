"""
Work with Jarvis-Patrick clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle

from scipy.spatial import distance_matrix
from scipy.special import comb

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################


def compute_k_nearest_neighbors(data, k):
    """
    Computes the k-nearest neighbors for each point in the dataset.
    
    Arguments:
    - data: numpy array of shape (n_samples, n_features)
    - k: int, number of nearest neighbors to find
    
    Returns:
    - neighbors: list of lists, where each sublist contains indices of the k-nearest neighbors for each point
    """
    dists = distance_matrix(data, data)
    neighbors = [np.argsort(dists[i])[1:k+1] for i in range(len(data))]
    return neighbors

def create_snn_graph(neighbors, smin):
    """
    Constructs the Shared Nearest Neighbor (SNN) graph.

    Arguments:
    - neighbors: list of lists, where each sublist contains the indices of the k-nearest neighbors for each point
    - smin: int, the minimum number of shared neighbors required to consider two points as connected in the graph

    Returns:
    - adjacency_matrix: a binary matrix indicating edges in the SNN graph
    """
    num_points = len(neighbors)
    adjacency_matrix = np.zeros((num_points, num_points), dtype=int)

    for i in range(num_points):
        for j in range(i + 1, num_points):
            shared_neighbors = set(neighbors[i]).intersection(neighbors[j])
            if len(shared_neighbors) >= smin:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1  # Since the graph is undirected

    return adjacency_matrix


def dbscan_snn(snn_graph, min_pts):
    """
    Clusters points based on the SNN graph using a DBSCAN-like approach.
    
    Arguments:
    - snn_graph: numpy array, adjacency matrix of the SNN graph
    - min_pts: int, minimum number of points in a neighborhood to form a cluster
    
    Returns:
    - labels: numpy array, cluster labels for each point
    """
    num_points = len(snn_graph)
    labels = -np.ones(num_points, dtype=int)
    cluster_id = 0

    def expand_cluster(point_index, cluster_id):
        points_to_visit = [point_index]
        while points_to_visit:
            current_point = points_to_visit.pop()
            if labels[current_point] == -1:  # If the point is not yet visited
                labels[current_point] = cluster_id
                # Get all neighbors of the current point
                neighbors = np.where(snn_graph[current_point] == 1)[0]
                if len(neighbors) >= min_pts:
                    points_to_visit.extend(neighbors)
    
    for point in range(num_points):
        if labels[point] == -1:  # If the point is not yet visited
            # Check if the point has enough neighbors to start a new cluster
            if np.sum(snn_graph[point]) >= min_pts:
                expand_cluster(point, cluster_id)
                cluster_id += 1

    return labels

def calculate_ari(labels_true, labels_pred):
    """
    Calculate the Adjusted Rand Index (ARI) considering labels might include noise points labeled as -1.
    """
    import numpy as np
    from scipy.special import comb

    # Filter out noise points
    valid_indices = (labels_true != -1) & (labels_pred != -1)
    labels_true = labels_true[valid_indices]
    labels_pred = labels_pred[valid_indices]

    # Handle the case where no valid indices exist
    if len(labels_true) == 0 or len(labels_pred) == 0:
        return 0.0

    # Create a contingency table
    max_true = labels_true.max() + 1
    max_pred = labels_pred.max() + 1
    contingency = np.zeros((max_true, max_pred), dtype=int)
    for i in range(len(labels_true)):
        contingency[labels_true[i], labels_pred[i]] += 1

    # Calculate combinations for each element in the contingency table
    comb_contingency = np.sum([comb(n, 2) for n in contingency.flatten()])
    comb_rows = np.sum([comb(n, 2) for n in np.sum(contingency, axis=1)])
    comb_cols = np.sum([comb(n, 2) for n in np.sum(contingency, axis=0)])
    total_comb = comb(len(labels_true), 2)

    # Expected index (by chance alignment)
    expected_index = comb_rows * comb_cols / total_comb if total_comb > 0 else 0

    # Adjusted Rand Index
    max_index = (comb_rows + comb_cols) / 2
    if max_index == expected_index:
        return 0.0
    else:
        ari = (comb_contingency - expected_index) / (max_index - expected_index)
    return ari

def compute_sse(data, labels):
    """
    Computes the Sum of Squared Errors (SSE) for the clusters.
    
    Arguments:
    - data: numpy array of data points
    - labels: numpy array of cluster labels for each data point
    
    Returns:
    - sse: float, the sum of squared errors for all clusters
    """
    unique_labels = set(labels) - {-1}  # Exclude noise points
    sse = 0
    for label in unique_labels:
        cluster_data = data[labels == label]
        if cluster_data.size == 0:
            continue
        centroid = np.mean(cluster_data, axis=0)
        sse += np.sum((cluster_data - centroid) ** 2)
    return sse



def jarvis_patrick(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    """
    Implementation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'k', 'smin', There could be others.
    - params_dict['k']: the number of nearest neighbors to consider. This determines the size of the neighborhood used to assess the similarity between datapoints. Choose values in the range 3 to 8
    - params_dict['smin']:  the minimum number of shared neighbors to consider two points in the same cluster.
       Choose values in the range 4 to 10.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index

    Notes:
    - the nearest neighbors can be bidirectional or unidirectional
    - Bidirectional: if point A is a nearest neighbor of point B, then point B is a nearest neighbor of point A).
    - Unidirectional: if point A is a nearest neighbor of point B, then point B is not necessarily a nearest neighbor of point A).
    - In this project, only consider unidirectional nearest neighboars for simplicity.
    - The metric  used to compute the the k-nearest neighberhood of all points is the Euclidean metric
    """

    # Extract parameters
    k = params_dict.get('k', 5)  # Default to 5 if not specified
    smin = params_dict.get('smin', 4)  # Default to 4 if not specified
    min_pts = params_dict.get('min_pts', 3)  # Assuming a default if not in params

    # Compute nearest neighbors
    neighbors = compute_k_nearest_neighbors(data, k)

    # Create SNN graph
    snn_graph = create_snn_graph(neighbors, smin)

    # Cluster data using DBSCAN-like approach on SNN graph
    computed_labels = dbscan_snn(snn_graph, min_pts)

    # Compute SSE
    SSE = compute_sse(data, computed_labels)

    # Compute ARI
    ARI = calculate_ari(labels, computed_labels)

    return computed_labels, SSE, ARI



import numpy as np
import matplotlib.pyplot as plt

def jarvis_patrick_clustering():
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    # Load the data
    data = np.load('question1_cluster_data.npy')  # Adjust the path as needed
    labels = np.load('question1_cluster_labels.npy')  # Adjust the path as needed

    # Initialize answers dictionary
    answers = {}

    # Define parameters for the study
    k_values = np.linspace(3, 8, 6, dtype=int)  # Values for k
    smin_values = np.linspace(4, 10, 7, dtype=int)  # Values for smin
    groups = {}

    # Perform clustering for each combination of parameters
    for k in k_values:
        for smin in smin_values:
            params_dict = {'k': k, 'smin': smin}
            computed_labels, sse, ari = jarvis_patrick(data[:1000], labels[:1000], params_dict)
            groups[(k, smin)] = {'ARI': ari, 'SSE': sse, 'labels': computed_labels}

    # Analyzing results to find the best and worst parameters
    best_params = max(groups, key=lambda x: groups[x]['ARI'])
    worst_params = min(groups, key=lambda x: groups[x]['SSE'])

    # Plot for the best ARI
    best_ari_data = data[:1000][groups[best_params]['labels'] != -1]
    plt.figure()
    plot_ARI = plt.scatter(best_ari_data[:, 0], best_ari_data[:, 1], c='blue')
    plt.title('Cluster with Largest ARI')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()

    # Plot for the worst SSE
    worst_sse_data = data[:1000][groups[worst_params]['labels'] != -1]
    plt.figure()
    plot_SSE = plt.scatter(worst_sse_data[:, 0], worst_sse_data[:, 1], c='red')
    plt.title('Cluster with Smallest SSE')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()

    # Assign values to the answers dictionary
    answers["jarvis_patrick_function"] = jarvis_patrick
    answers["cluster parameters"] = groups
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Statistical analysis for ARIs and SSEs
    ari_values = [group['ARI'] for group in groups.values()]
    sse_values = [group['SSE'] for group in groups.values()]
    answers["mean_ARIs"] = np.mean(ari_values)
    answers["std_ARIs"] = np.std(ari_values)
    answers["mean_SSEs"] = np.mean(sse_values)
    answers["std_SSEs"] = np.std(sse_values)

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
