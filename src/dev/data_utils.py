#!/usr/bin/env python3
"""
Data utilities for loading and managing graph datasets
"""

import os
import networkx as nx
import numpy as np
from pathlib import Path

class DataUtils:
    """Utility class for handling graph datasets"""
    
    def __init__(self, data_dir="../data"):
        """
        Initialize DataUtils with data directory
        Args:
            data_dir (str): Path to data directory
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_karate_club(self):
        """
        Load Zachary's Karate Club dataset
        Returns:
            nx.Graph: The Karate Club network
            dict: Ground truth community labels
        """
        print("Loading Zachary's Karate Club dataset...")
        
        # Load the dataset
        G = nx.karate_club_graph()
        
        # Get ground truth communities
        # In the original dataset, nodes are labeled as either "Mr. Hi" (0) or "Officer" (1)
        communities = {}
        for node in G.nodes():
            # Convert string labels to numeric (0 for "Mr. Hi", 1 for "Officer")
            communities[node] = 0 if G.nodes[node]['club'] == 'Mr. Hi' else 1
        
        print(f"Loaded Karate Club network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G, communities
    
    def save_graph_info(self, G, communities, filename="karate_club_info.txt"):
        """
        Save graph information to a text file
        Args:
            G (nx.Graph): The graph to analyze
            communities (dict): Community labels
            filename (str): Output filename
        """
        output_path = self.data_dir / filename
        
        with open(output_path, 'w') as f:
            f.write("Graph Information:\n")
            f.write(f"Number of nodes: {G.number_of_nodes()}\n")
            f.write(f"Number of edges: {G.number_of_edges()}\n")
            f.write("\nNode degrees:\n")
            for node in sorted(G.nodes()):
                f.write(f"Node {node}: degree {G.degree(node)}, community {communities[node]}\n")
            
            f.write("\nEdge list:\n")
            for edge in sorted(G.edges()):
                f.write(f"{edge[0]} -- {edge[1]}\n")
        
        print(f"Graph information saved to {output_path}")
    
    def get_community_sizes(self, communities):
        """
        Get the size of each community
        Args:
            communities (dict): Community labels
        Returns:
            dict: Community sizes
        """
        community_sizes = {}
        for node, community in communities.items():
            community_sizes[community] = community_sizes.get(community, 0) + 1
        return community_sizes 