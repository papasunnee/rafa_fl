"""
har_loader.py — UCI HAR dataset loading and preprocessing (Section 4.2.1).

Loads the standard UCI HAR train/test split, applies StandardScaler fitted on
the training set only (no leakage), and zero-indexes labels for CrossEntropyLoss.
"""

import os
import numpy as np


def load_har(dataset_root):
    """
    Load UCI HAR. Expects the standard directory layout:
        <root>/train/X_train.txt, <root>/train/y_train.txt
        <root>/test/X_test.txt,  <root>/test/y_test.txt
    Returns (X_train, y_train, X_test, y_test) as NumPy arrays,
    with features standardised (train statistics) and labels zero-indexed.
    """
    def _load(split, xname, yname):
        X = np.loadtxt(os.path.join(dataset_root, split, xname))
        y = np.loadtxt(os.path.join(dataset_root, split, yname)).astype(int)
        return X, y

    X_train, y_train = _load("train", "X_train.txt", "y_train.txt")
    X_test, y_test = _load("test", "X_test.txt", "y_test.txt")

    # Standardise using training statistics only (prevent leakage)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Labels 1..6 -> 0..5
    y_train -= 1
    y_test -= 1

    return (X_train.astype(np.float32), y_train,
            X_test.astype(np.float32), y_test)


def stratified_val_split(indices, labels, val_fraction, rng):
    """
    Split a client's assigned indices into train/val, stratified by class,
    performed AFTER Dirichlet partitioning so the val set reflects the same
    non-IID distribution as train (Section 4.2.1).
    """
    indices = np.asarray(indices)
    train_idx, val_idx = [], []
    for c in np.unique(labels[indices]):
        cls = indices[labels[indices] == c]
        rng.shuffle(cls)
        n_val = max(1, int(len(cls) * val_fraction)) if len(cls) > 1 else 0
        val_idx.extend(cls[:n_val])
        train_idx.extend(cls[n_val:])
    return np.array(train_idx), np.array(val_idx)
