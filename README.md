# Spectral Graph Clustering for Community Detection

This project implements spectral graph theory for community detection in social networks, using the Zachary's Karate Club dataset as a case study.

## Dataset: Zachary's Karate Club

The Zachary's Karate Club dataset is a classic social network dataset that represents the social relationships between members of a university karate club. This dataset is particularly valuable for community detection because:

1. **Real-World Social Network**: Captures actual social interactions and relationships
2. **Known Ground Truth**: The true community structure is known (split into two groups)
3. **Well-Studied**: Extensive research has been done on this dataset
4. **Manageable Size**: 34 nodes and 78 edges, making it perfect for learning and testing

### Dataset Characteristics:
- **Nodes**: 34 members of the karate club
- **Edges**: 78 social interactions between members
- **Communities**: 2 distinct groups (Mr. Hi's group and Officer's group)
- **Time Period**: Observations over 2 years
- **Split Reason**: Dispute between the instructor (Mr. Hi) and the administrator (Officer)

### Why This Dataset?
- Perfect for learning spectral clustering
- Small enough to understand completely
- Large enough to be meaningful
- Has a clear community structure
- Well-documented ground truth

## Results and Visualization

After running the spectral clustering algorithm, you'll get several visualizations in the `results` directory:

Sample results: 

image.png

1. **Community Detection Results**:
   - True communities vs. predicted communities
   - Spectral embedding visualization
   - Decision boundary in spectral space

2. **Eigenvector Analysis**:
   - First eigenvector values by node
   - Distribution of eigenvector values
   - Community separation visualization

3. **Example Results**:
![Community Detection Results](results/example_results.png)
*Note: Replace with your actual results image*

The visualizations show:
- How well the algorithm identifies the two communities
- The natural separation in the spectral embedding
- The effectiveness of the spectral clustering approach

## What is Spectral Graph Theory?

Spectral graph theory is a powerful mathematical framework that studies the relationship between the structural properties of graphs and the eigenvalues/eigenvectors of matrices associated with these graphs. The key matrices used are:

1. **Adjacency Matrix (A)**: Represents direct connections between nodes
2. **Degree Matrix (D)**: Diagonal matrix of node degrees
3. **Laplacian Matrix (L)**: L = D - A, captures the graph's connectivity structure
4. **Normalized Laplacian**: L_norm = I - D^(-1/2)AD^(-1/2), provides better spectral properties

## How Spectral Clustering Works

1. **Graph Representation**: Convert the network into a graph with nodes and edges
2. **Laplacian Computation**: Compute the normalized Laplacian matrix
3. **Spectral Decomposition**: Find eigenvalues and eigenvectors of the Laplacian
4. **Dimensionality Reduction**: Use the first k eigenvectors (excluding the constant vector)
5. **Community Detection**: Cluster nodes in the reduced spectral space

## Real-World Applications

### 1. Social Network Analysis
- **Community Detection**: Identifying groups of closely connected users
- **Influence Analysis**: Finding key influencers and their communities
- **Recommendation Systems**: Suggesting connections based on community structure

### 2. Computer Vision
- **Image Segmentation**: Partitioning images into meaningful regions
- **Object Recognition**: Grouping similar visual features
- **Face Clustering**: Organizing face images into identity groups

### 3. Bioinformatics
- **Protein Interaction Networks**: Identifying functional modules
- **Gene Expression Analysis**: Clustering genes with similar expression patterns
- **Disease Pathway Analysis**: Understanding disease mechanisms

### 4. Network Security
- **Anomaly Detection**: Identifying unusual patterns in network traffic
- **Bot Detection**: Finding clusters of automated behavior
- **Fraud Detection**: Discovering coordinated fraudulent activities

### 5. Transportation Networks
- **Traffic Flow Analysis**: Understanding community structure in road networks
- **Public Transport Optimization**: Identifying natural service areas
- **Infrastructure Planning**: Planning based on community connectivity

## Advantages of Spectral Clustering

1. **Global Structure**: Captures the overall structure of the network
2. **Robustness**: Less sensitive to noise and local variations
3. **Theoretical Foundation**: Well-grounded in mathematical theory
4. **Flexibility**: Can handle various types of networks and data
5. **Quality Guarantees**: Provides theoretical guarantees for cluster quality

## Project Structure

```
.
├── src/
│   ├── dev/           # Development code
│   ├── test/          # Test files
│   └── data/          # Data files
├── results/           # Output results with timestamps
└── requirements.txt   # Project dependencies
```

## Features

- Graph construction and manipulation using NetworkX
- Spectral clustering implementation using NumPy and SciPy
- Visualization of results using Matplotlib
- Detailed debugging information and comments
- Test-driven development approach

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python src/dev/main.py
```

2. Run tests:
```bash
pytest src/test/
```

## Learning Resources

1. **Spectral Graph Theory**:
   - [Spectral Graph Theory by Fan Chung](https://www.math.ucsd.edu/~fan/research/revised.html)
   - [Spectral Clustering Tutorial](https://www.cs.cmu.edu/~aarti/Class/10701/readings/Luxburg06_TR.pdf)

2. **Community Detection**:
   - [Community Detection in Networks](https://arxiv.org/abs/0906.0612)
   - [Spectral Clustering and its Applications](https://www.cs.cmu.edu/~aarti/Class/10701/readings/Luxburg06_TR.pdf)

3. **Network Analysis**:
   - [Network Science by Albert-László Barabási](http://networksciencebook.com/)
   - [Social Network Analysis: Methods and Applications](https://www.cambridge.org/core/books/social-network-analysis/8E3B4D1F6A5F4B5F8E3B4D1F6A5F4B5F)

## Future Enhancements

1. Support for weighted graphs
2. Dynamic community detection
3. Multi-scale community analysis
4. Integration with other clustering methods
5. Real-time community detection
6. Support for directed graphs
7. Enhanced visualization capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
