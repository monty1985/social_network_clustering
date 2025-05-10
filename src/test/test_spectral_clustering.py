#!/usr/bin/env python3
"""
Test suite for spectral clustering implementation
"""

import pytest
import numpy as np
import networkx as nx
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dev.main import (
    compute_laplacian,
    spectral_embedding,
    cluster_nodes,
    evaluate_clustering
)
from dev.data_utils import DataUtils

@pytest.fixture
def karate_data():
    """Fixture to load Karate Club dataset"""
    data_utils = DataUtils()
    G, communities = data_utils.load_karate_club()
    return G, communities

def test_compute_laplacian(karate_data):
    """Test Laplacian matrix computation"""
    G, _ = karate_data
    L = compute_laplacian(G)
    
    # Check matrix dimensions
    assert L.shape == (34, 34)  # Karate Club has 34 nodes
    
    # Check if Laplacian is symmetric
    assert np.allclose(L, L.T)
    
    # Check if diagonal elements are 1 (for normalized Laplacian)
    assert np.allclose(np.diag(L), 1.0)

def test_spectral_embedding(karate_data):
    """Test spectral embedding computation"""
    G, _ = karate_data
    L = compute_laplacian(G)
    embedding = spectral_embedding(L, n_clusters=2)
    
    # Check embedding dimensions
    assert embedding.shape == (34, 2)  # 34 nodes, 2 dimensions
    
    # Check if embedding vectors are normalized
    assert np.allclose(np.sum(embedding**2, axis=0), 1.0)

def test_cluster_nodes(karate_data):
    """Test node clustering"""
    G, _ = karate_data
    L = compute_laplacian(G)
    embedding = spectral_embedding(L)
    labels = cluster_nodes(embedding)
    
    # Check if labels are binary (0 or 1)
    assert set(np.unique(labels)) == {0, 1}
    
    # Check if number of labels matches number of nodes
    assert len(labels) == G.number_of_nodes()

def test_end_to_end(karate_data):
    """Test the entire pipeline"""
    G, true_communities = karate_data
    L = compute_laplacian(G)
    embedding = spectral_embedding(L)
    pred_labels = cluster_nodes(embedding)
    
    # Convert true communities to array
    true_labels = np.array([true_communities[node] for node in sorted(G.nodes())])
    
    # Evaluate clustering
    ari = evaluate_clustering(true_labels, pred_labels)
    
    # Check if ARI is reasonable (should be > 0.5 for good clustering)
    assert ari > 0.5, "Clustering quality is too low" 