# Spectral Graph Clustering Project

This project implements spectral graph clustering to analyze and visualize community structures in networks. The implementation focuses on learning spectral graph theory concepts through a hands-on approach.

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

- Spectral Graph Theory
- Graph Laplacian
- Community Detection
- K-means Clustering 