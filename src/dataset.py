"""Dataset loader and RBF kernel builder.

This module provides helpers to load MNIST dataset using mnist_datasets package and build an RBF kernel matrix
K_{ij} = exp(-||x_i - x_j||^2 / c^2) for the first `n` rows of the dataset.
"""
from typing import Tuple
import numpy as np
from mnist_datasets import MNISTLoader
import os


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


def build_kernel_from_mnist(n: int, c: float, train: bool = True) -> np.ndarray:
    """Helper: load first n rows from MNIST and build RBF kernel."""
    X = load_mnist(n=n, train=train)
    return rbf_kernel(X, c)
