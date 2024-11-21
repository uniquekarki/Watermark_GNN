# Watermarking Graph Neural Networks by Random Graphs

This repository implements the methodology described in the research paper "Watermarking Graph Neural Networks by Random Graphs". The approach embeds a unique identifier into Graph Neural Networks (GNNs) using Erdos-Renyi (ER) random graphs, ensuring model integrity and ownership protection.

## Features
- Generate ER graphs.
- Combine ER graphs with existing datasets: Cora and PubMed.
- Train a GNN model with the original and combined graph data.
- Test the model authenticity with the ER graph and original data.

## Repository Structure
    |-- models/                # Trained Models
    |-- notebooks/             # Jupyter Notebook Files  
    |   |-- base_model.ipynb   # GNN Experimentation and Exploration  
    |   |-- ER_Graphs.ipynb    # ER Graphs Experimentation and Exploration
    |-- scripts/               # Python Scripts  
    |   |-- base_gnn.py        # Main Program File  
    |   |-- run.log            # Program Log File  
    |   |-- watermark.py       # Code for Watermarking  
    |-- README.md              # Documentation  
    |-- requirements.txt       # List of Dependencies

## References
    X. Zhao, H. Wu and X. Zhang, "Watermarking Graph Neural Networks by Random Graphs," 2021 9th International Symposium on Digital Forensics and Security (ISDFS), Elazig, Turkey, 2021, pp. 1-6, doi: 10.1109/ISDFS52919.2021.9486352. keywords: {Training;Digital forensics;Watermarking;Predictive models;Graph neural networks;Data models;Robustness;Watermarking;deep learning;graph neural networks;random graph},