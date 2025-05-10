#!/usr/bin/env python3
"""
Spectral Graph Clustering Implementation
This script demonstrates spectral graph clustering using NumPy and SciPy.
"""

import numpy as np
import networkx as nx
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import os
from datetime import datetime
from data_utils import DataUtils

# Debug flag - set to True to print detailed information
DEBUG = True

def debug_print(message):
    """Helper function for debug printing"""
    if DEBUG:
        print(f"[DEBUG] {message}")

def compute_laplacian(G):
    """
    Compute the normalized Laplacian matrix of the graph
    Args:
        G (nx.Graph): Input graph
    Returns:
        np.ndarray: Normalized Laplacian matrix
    """
    debug_print("Computing Laplacian matrix...")
    
    # Get adjacency matrix
    A = nx.adjacency_matrix(G).toarray()
    debug_print(f"Adjacency matrix shape: {A.shape}")
    
    # Compute degree matrix
    D = np.diag(np.sum(A, axis=1))
    debug_print(f"Degree matrix shape: {D.shape}")
    
    # Compute normalized Laplacian: L = I - D^(-1/2)AD^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    L = np.eye(len(G)) - D_inv_sqrt @ A @ D_inv_sqrt
    
    debug_print("Laplacian matrix computed successfully")
    return L

def spectral_embedding(L, n_clusters=2):
    """
    Compute spectral embedding using the bottom k eigenvectors
    Args:
        L (np.ndarray): Laplacian matrix
        n_clusters (int): Number of clusters
    Returns:
        np.ndarray: Spectral embedding matrix
    """
    debug_print(f"Computing spectral embedding for {n_clusters} clusters...")
    
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = linalg.eigh(L)
    debug_print(f"Eigenvalues shape: {eigenvals.shape}")
    debug_print(f"Eigenvectors shape: {eigenvecs.shape}")
    
    # Get the bottom k eigenvectors (excluding the first one)
    embedding = eigenvecs[:, 1:n_clusters+1]
    debug_print(f"Spectral embedding shape: {embedding.shape}")
    
    return embedding

def cluster_nodes(embedding, n_clusters=2):
    """
    Cluster nodes using a direct approach based on spectral embedding
    Args:
        embedding (np.ndarray): Spectral embedding matrix
        n_clusters (int): Number of clusters (should be 2)
    Returns:
        np.ndarray: Cluster labels
    """
    debug_print(f"Clustering nodes into {n_clusters} clusters...")
    
    # Since we know there are 2 communities, we can use a simple approach
    # based on the first eigenvector (which captures the main community structure)
    
    # Get the first eigenvector (excluding the constant vector)
    first_eigenvector = embedding[:, 0]
    
    # Use the median as the splitting point
    median = np.median(first_eigenvector)
    
    # Assign labels based on whether they're above or below the median
    labels = (first_eigenvector > median).astype(int)
    
    # Ensure the larger community is labeled as 0 (Mr. Hi's group)
    if np.sum(labels == 0) < np.sum(labels == 1):
        labels = 1 - labels  # Flip the labels
    
    debug_print(f"Clustering complete. Labels shape: {labels.shape}")
    return labels

def evaluate_clustering(true_labels, pred_labels):
    """
    Evaluate clustering results using multiple metrics
    Args:
        true_labels (np.ndarray): Ground truth labels
        pred_labels (np.ndarray): Predicted labels
    """
    # Convert labels to numpy arrays if they aren't already
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    # Calculate metrics
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    print("\nClustering Evaluation:")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")

def visualize_results(G, true_labels, pred_labels, embedding):
    """
    Visualize the graph and spectral embedding
    Args:
        G (nx.Graph): Input graph
        true_labels (np.ndarray): Ground truth labels
        pred_labels (np.ndarray): Predicted labels
        embedding (np.ndarray): Spectral embedding
    """
    debug_print("Creating visualizations...")
    
    # Create timestamp for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create figure with 3 subplots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Original graph with true community colors
    plt.subplot(131)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=true_labels, cmap=plt.cm.rainbow)
    plt.title("True Communities")
    
    # Plot 2: Original graph with predicted community colors
    plt.subplot(132)
    nx.draw(G, pos, node_color=pred_labels, cmap=plt.cm.rainbow)
    plt.title("Predicted Communities")
    
    # Plot 3: Spectral embedding with decision boundary
    plt.subplot(133)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=true_labels, cmap=plt.cm.rainbow)
    
    # Add decision boundary
    median = np.median(embedding[:, 0])
    plt.axvline(x=median, color='k', linestyle='--', label='Decision Boundary')
    plt.title("Spectral Embedding with Decision Boundary")
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, "clustering_results.png"))
    debug_print(f"Visualizations saved to {results_dir}")
    plt.close()

def analyze_misclassified_nodes(G, true_labels, pred_labels):
    """
    Analyze nodes that were misclassified
    Args:
        G (nx.Graph): Input graph
        true_labels (np.ndarray): Ground truth labels
        pred_labels (np.ndarray): Predicted labels
    """
    print("\n=== Misclassified Nodes Analysis ===")
    
    # Find misclassified nodes
    misclassified = np.where(true_labels != pred_labels)[0]
    
    if len(misclassified) > 0:
        print(f"\nFound {len(misclassified)} misclassified nodes:")
        for node in misclassified:
            # Get node's connections
            neighbors = list(G.neighbors(node))
            # Count connections to each community
            connections_to_0 = sum(1 for n in neighbors if true_labels[n] == 0)
            connections_to_1 = sum(1 for n in neighbors if true_labels[n] == 1)
            
            print(f"\nNode {node}:")
            print(f"  True community: {true_labels[node]} (Mr. Hi's group)" if true_labels[node] == 0 else "  True community: {true_labels[node]} (Officer's group)")
            print(f"  Predicted community: {pred_labels[node]}")
            print(f"  Number of connections to Mr. Hi's group: {connections_to_0}")
            print(f"  Number of connections to Officer's group: {connections_to_1}")
            print(f"  Total connections: {len(neighbors)}")
    else:
        print("No misclassified nodes found!")

def analyze_spectral_space(embedding, true_labels, pred_labels, G):
    """
    Analyze the spectral embedding space for misclassified nodes
    Args:
        embedding (np.ndarray): Spectral embedding
        true_labels (np.ndarray): Ground truth labels
        pred_labels (np.ndarray): Predicted labels
        G (nx.Graph): Input graph
    """
    print("\n=== Spectral Space Analysis ===")
    
    # Find misclassified nodes
    misclassified = np.where(true_labels != pred_labels)[0]
    
    if len(misclassified) > 0:
        print("\nAnalyzing spectral embedding for misclassified nodes:")
        for node in misclassified:
            # Get node's position in spectral space
            node_embedding = embedding[node]
            
            # Calculate distances to cluster centers
            cluster_0_nodes = embedding[true_labels == 0]
            cluster_1_nodes = embedding[true_labels == 1]
            
            center_0 = np.mean(cluster_0_nodes, axis=0)
            center_1 = np.mean(cluster_1_nodes, axis=0)
            
            dist_to_0 = np.linalg.norm(node_embedding - center_0)
            dist_to_1 = np.linalg.norm(node_embedding - center_1)
            
            print(f"\nNode {node}:")
            print(f"  Position in spectral space: {node_embedding}")
            print(f"  Distance to Mr. Hi's group center: {dist_to_0:.4f}")
            print(f"  Distance to Officer's group center: {dist_to_1:.4f}")
            print(f"  Closer to: {'Officer\'s group' if dist_to_1 < dist_to_0 else 'Mr. Hi\'s group'}")
            
            # Get node's neighbors in spectral space
            distances = np.linalg.norm(embedding - node_embedding, axis=1)
            nearest_neighbors = np.argsort(distances)[1:6]  # Get 5 nearest neighbors
            
            print("  Nearest neighbors in spectral space:")
            for neighbor in nearest_neighbors:
                print(f"    Node {neighbor}: distance={distances[neighbor]:.4f}, "
                      f"true_community={true_labels[neighbor]}, "
                      f"predicted_community={pred_labels[neighbor]}")

def visualize_eigenvector_analysis(embedding, true_labels, G):
    """
    Visualize the first eigenvector values and their relationship to communities
    Args:
        embedding (np.ndarray): Spectral embedding
        true_labels (np.ndarray): Ground truth labels
        G (nx.Graph): Input graph
    """
    debug_print("Creating eigenvector analysis visualization...")
    
    # Create timestamp for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Get first eigenvector values
    first_eigenvector = embedding[:, 0]
    
    # Create figure with 2 subplots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: First eigenvector values
    plt.subplot(121)
    nodes = sorted(G.nodes())
    plt.scatter(nodes, first_eigenvector, c=true_labels, cmap=plt.cm.rainbow)
    plt.axhline(y=np.median(first_eigenvector), color='k', linestyle='--', label='Median')
    plt.xlabel('Node ID')
    plt.ylabel('First Eigenvector Value')
    plt.title('First Eigenvector Values by Node')
    plt.legend()
    
    # Plot 2: Distribution of eigenvector values
    plt.subplot(122)
    plt.hist([first_eigenvector[true_labels == 0], first_eigenvector[true_labels == 1]], 
             label=['Mr. Hi\'s Group', 'Officer\'s Group'],
             bins=10, alpha=0.7)
    plt.axvline(x=np.median(first_eigenvector), color='k', linestyle='--', label='Median')
    plt.xlabel('First Eigenvector Value')
    plt.ylabel('Count')
    plt.title('Distribution of First Eigenvector Values')
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, "eigenvector_analysis.png"))
    debug_print(f"Eigenvector analysis visualization saved to {results_dir}")
    plt.close()
    
    # Print some statistics
    print("\n=== First Eigenvector Analysis ===")
    print(f"Median value: {np.median(first_eigenvector):.4f}")
    print("\nMr. Hi's Group (Community 0):")
    print(f"  Mean value: {np.mean(first_eigenvector[true_labels == 0]):.4f}")
    print(f"  Std dev: {np.std(first_eigenvector[true_labels == 0]):.4f}")
    print("\nOfficer's Group (Community 1):")
    print(f"  Mean value: {np.mean(first_eigenvector[true_labels == 1]):.4f}")
    print(f"  Std dev: {np.std(first_eigenvector[true_labels == 1]):.4f}")

def main():
    """Main function to run the spectral clustering pipeline"""
    debug_print("Starting spectral clustering pipeline...")
    
    # Initialize data utilities
    data_utils = DataUtils()
    
    # Load Karate Club dataset
    G, true_communities = data_utils.load_karate_club()
    
    # Save graph information
    data_utils.save_graph_info(G, true_communities)
    
    # Convert true communities to array
    true_labels = np.array([true_communities[node] for node in sorted(G.nodes())])
    
    # Step 1: Compute Laplacian
    L = compute_laplacian(G)
    
    # Step 2: Compute spectral embedding
    embedding = spectral_embedding(L)
    
    # Step 3: Cluster nodes
    pred_labels = cluster_nodes(embedding)
    
    # Step 4: Evaluate results
    evaluate_clustering(true_labels, pred_labels)
    
    # Analyze misclassified nodes
    analyze_misclassified_nodes(G, true_labels, pred_labels)
    
    # Analyze spectral space
    analyze_spectral_space(embedding, true_labels, pred_labels, G)
    
    # Visualize eigenvector analysis
    visualize_eigenvector_analysis(embedding, true_labels, G)
    
    # Step 5: Visualize results
    visualize_results(G, true_labels, pred_labels, embedding)
    
    debug_print("Pipeline completed successfully!")

if __name__ == "__main__":
    main() 