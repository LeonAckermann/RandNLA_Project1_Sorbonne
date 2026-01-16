"""Dataset loader and RBF kernel builder.

This module provides helpers to load MNIST dataset using mnist_datasets package and build an RBF kernel matrix
K_{ij} = exp(-||x_i - x_j||^2 / c^2) for the first `n` rows of the dataset.
"""
from typing import Tuple
import numpy as np
from mnist_datasets import MNISTLoader
import os
import pandas as pd
import requests
import zipfile
import io

# --- Configuration Constants ---
MSD_TRAIN_SIZE = 463715
MSD_TEST_SIZE = 51630
MSD_UCI_URL = "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip"
MSD_INTERNAL_FILENAME = "YearPredictionMSD.txt"  # Filename inside the UCI zip

def load_mnist(n: int = None, dtype=np.float64, train: bool = True) -> np.ndarray:
    """Load MNIST dataset and return a dense array of shape (n_rows, 784).

    Args:
        n: number of rows to keep (first n). If None, load all (60000 for train, 10000 for test).
        dtype: data type for the array.
        train: if True, load training set; else test set.
    Returns:
        X: dense numpy array (n x 784)
    """
    loader = MNISTLoader()
    images, _ = loader.load(train=train)
    # Flatten images: each is 28x28, so reshape to (n_samples, 784)
    X = images.reshape(images.shape[0], -1).astype(dtype)
    if n is not None:
        X = X[:n, :]
    return X

def _ensure_msd_exists(path: str) -> None:
    """Internal helper: Download YearPredictionMSD if not found at path."""
    if os.path.exists(path):
        return

    print(f"Dataset not found at '{path}'. Downloading from UCI...")
    
    try:
        response = requests.get(MSD_UCI_URL, stream=True)
        response.raise_for_status()
        
        # Open the zip file from memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # The zip usually contains a single file: 'YearPredictionMSD.txt'
            if MSD_INTERNAL_FILENAME not in z.namelist():
                raise ValueError(f"Unexpected zip content. Expected {MSD_INTERNAL_FILENAME}")
            
            # Extract to the directory where 'path' is located
            extract_dir = os.path.dirname(path) or "."
            z.extract(MSD_INTERNAL_FILENAME, path=extract_dir)
            
            # If the user provided a custom filename (e.g. 'data/msd.csv') 
            # but the zip contained 'YearPredictionMSD.txt', we rename it to match 'path'.
            extracted_path = os.path.join(extract_dir, MSD_INTERNAL_FILENAME)
            if os.path.abspath(extracted_path) != os.path.abspath(path):
                os.rename(extracted_path, path)
                
        print("Download and extraction complete.")

    except Exception as e:
        print(f"Failed to download dataset: {e}")
        raise


def load_year_prediction_msd(n: int = None, dtype=np.float64, train: bool = True, path: str = './datasets/YearPredictionMSD.txt') -> np.ndarray:
    """Load Year Prediction MSD dataset. Downloads automatically if missing.

    The dataset contains 90 audio features. The first column (Year) is discarded.

    Args:
        n: number of rows to load. If None, load all available for the specific split.
        dtype: data type for the array.
        train: if True, load the first 463,715 samples; else load the last 51,630.
        path: File path where the dataset is (or should be) stored (default: './datasets/YearPredictionMSD.txt').

    Returns:
        X: dense numpy array (n x 90)
    """
    # 1. Ensure file exists (Download if missing)
    _ensure_msd_exists(path)

    # 2. Logic to handle the fixed split defined by the dataset creators
    if train:
        skip_rows = 0
        max_rows = MSD_TRAIN_SIZE
    else:
        skip_rows = MSD_TRAIN_SIZE
        max_rows = MSD_TEST_SIZE

    # Determine how many rows to actually read
    if n is not None:
        nrows = min(n, max_rows)
    else:
        nrows = max_rows

    # 3. Read CSV
    # No header in original file. Column 0 is Year, Columns 1-90 are features.
    df = pd.read_csv(
        path, 
        header=None, 
        skiprows=skip_rows, 
        nrows=nrows
    )

    # Drop the first column (the target Year) to get just the features
    X = df.iloc[:, 1:].values.astype(dtype)
    
    return X


def rbf_kernel(X: np.ndarray, c: float) -> np.ndarray:
    """Compute the dense RBF kernel matrix K where
    K_{ij} = exp(-||x_i-x_j||^2 / c^2).

    Args:
        X: (n x d) data array
        c: bandwidth parameter (float)

    Returns:
        K: (n x n) dense kernel matrix
    """
    X = np.asarray(X)
    sq_norms = np.sum(X * X, axis=1)
    # pairwise squared distances: ||xi-xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi^T xj
    D2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
    # numerical cleanup
    D2 = np.maximum(D2, 0.0)
    K = np.exp(-D2 / (c * c))
    return K


