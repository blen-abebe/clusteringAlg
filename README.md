# Markov State Model Analysis of Rheb GTPase Conformational Dynamics

## Overview

This project analyzes the conformational dynamics of the Rheb GTPase protein using Markov State Models (MSMs). The workflow includes data loading, feature engineering, clustering (HDBSCAN and k-means), MSM construction, and model validation using PyEMMA.

## Goals

- **Load and preprocess** tilt and rotation angle data for Rheb GTPase.
- **Visualize** the distribution and relationship of tilt and rotation angles.
- **Cluster** the conformational space using both HDBSCAN (density-based) and k-means (partition-based) algorithms.
- **Build MSMs** from the discrete trajectories obtained from clustering.
- **Validate MSMs** using implied timescales and Chapman-Kolmogorov (CK) tests.
- **Compare** the performance and interpretability of HDBSCAN and k-means clustering for MSM construction.
- **Estimate Bayesian MSMs** to quantify uncertainty in model predictions.

## Main Steps

1. **Data Loading & Preprocessing**
   - Load tilt and rotation angle data from text files.
   - Convert angles to radians and create circular features for clustering.

2. **Visualization**
   - Plot histograms and scatter plots to explore the data.

3. **Clustering**
   - Use HDBSCAN to find clusters without specifying the number of clusters.
   - Use k-means clustering (with k=100) to discretize the conformational space.

4. **MSM Construction**
   - Assign each frame to a cluster to create discrete trajectories.
   - Estimate MSMs using PyEMMA for both clustering methods.

5. **Model Validation**
   - Plot implied timescales to select an appropriate lag time.
   - Perform CK tests to validate the Markovianity of the models.
   - Use Bayesian MSMs to visualize uncertainty (shaded regions in CK test plots).

6. **Interpretation**
   - Analyze stationary distributions, transition matrices, and timescales.
   - Compare the results from HDBSCAN and k-means clustering.

## Files

- `markovModel.ipynb`: Main Jupyter notebook containing all code, plots, and analysis.
- `Rheb_GTPtiltangles.txt`, `Rheb_GTProtationangles.txt`: Input data files with angle measurements.

## Requirements

- Python 3.10+
- numpy
- matplotlib
- hdbscan
- pyemma

## How to Run

1. Install dependencies (preferably in a virtual environment).
2. Place the data files in the same directory as the notebook.
3. Open and run `markovModel.ipynb` in Jupyter or VS Code.

## Results

- Visualizations of angle distributions and clusters.
- MSMs built from both HDBSCAN and k-means clusters.
- CK test plots (with and without Bayesian uncertainty).
- Insights into the metastable states and kinetics of Rheb GTPase.
