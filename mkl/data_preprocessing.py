
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def normalization(X):
    return (X.T / torch.norm(X, dim=1, p=2)).T



def load_Y():
    csvname = './data/labels.csv'

    y = np.loadtxt(csvname, delimiter=',', dtype=str, skiprows=1)

    # y.shape

    y_train = y[:, 1]
    # y_train
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)

    y_float = y_encoded.astype(int)

    return y_float


def load_X():
    df = pd.read_csv("./data/data.csv")

    df_drop = df.iloc[:, 1:]

    X_train = df_drop.to_numpy()

    return X_train



def preprocess(X_train, y_float):
    y = y_float.reshape(-1, 1)

    label_counts = np.bincount(y.flatten())

    most_common_labels = np.argsort(label_counts)[-2:]

    filtered_indices = np.isin(y.flatten(), most_common_labels)
    X_filtered = X_train[filtered_indices]
    y_filtered = y[filtered_indices]

    X_train = X_filtered
    y_train = y_filtered.flatten()

    return X_train, y_train




def class_redistribution(X_train, y_train):
    print("Class distribution before SMOTE:", Counter(y_train))
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print("Class distribution after SMOTE:", Counter(y_resampled))
    return (X_resampled, y_resampled)


def pca(X_resampled):
    def find_optimal_num_components(X, threshold=0.99):
        pca = PCA()
        pca.fit(X)
        explained_variance = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance)
        num_components = np.argmax(cumulative_explained_variance >= threshold) + 1
        return num_components

    num_components = find_optimal_num_components(X_resampled)

    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(X_resampled)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    return X_pca

def load_and_process():
    import os
    current_directory = os.getcwd()
    print(current_directory)

    X = load_X()
    Y = load_Y()

    X, Y = preprocess(X, Y)
    # X, Y = class_redistribution(X, Y)
    X = pca(X)

    import numpy as np

    # Function to add random noise to the data
    def add_noise(X, noise_level=0.6):
        # Generate random noise with the same shape as X
        noise = np.random.normal(loc=0, scale=noise_level, size=X.shape)
        # Add noise to the data
        X_noisy = X + noise
        return X_noisy

    # Example usage
    # Assuming X is your input data and Y is the corresponding labels
    X = add_noise(X)
    # Example usage
    # Assuming X is your gene expression data
    X_augmented = gene_dropout(X)
    X_augmented = random_shifts(X_augmented)
    X_augmented = scale_data(X_augmented)

    X_augmented = gene_set_perturbation(X_augmented, gene_sets=[[0, 1, 2], [3, 4, 5]])
    X_augmented, Y = generate_synthetic_data(X_augmented, Y)
    X = drop_features(X_augmented)
    X = normalization(X)

    save_X_and_Y(X, Y)

    return X, Y


import numpy as np
from sklearn.decomposition import PCA

# Function to simulate gene dropout
def gene_dropout(X, dropout_rate=0.1):
    # Copy the original data
    X_augmented = X.copy()
    # Get the number of genes to drop
    num_genes = int(dropout_rate * X.shape[1])
    # Randomly select genes to drop for each sample
    for i in range(X.shape[0]):
        dropout_indices = np.random.choice(X.shape[1], size=num_genes, replace=False)
        X_augmented[i, dropout_indices] = 0  # Set selected genes to zero
    return X_augmented

def scale_data(X):
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X_scaled

# Function to perturb gene sets
def gene_set_perturbation(X, gene_sets, perturbation_strength=0.1):
    # Copy the original data
    X_augmented = X.copy()
    # Perturb gene sets by adding random noise
    for gene_set in gene_sets:
        num_genes = len(gene_set)
        noise = np.random.normal(loc=0, scale=perturbation_strength, size=X.shape[0])
        for i, gene_index in enumerate(gene_set):
            X_augmented[:, gene_index] += noise * (i + 1)  # Increase noise for genes within the set
    return X_augmented

# Function to drop some features
def drop_features(X, drop_rate=0.1):
    # Calculate the number of features to drop
    num_features = int(drop_rate * X.shape[1])
    # Randomly select features to drop
    drop_indices = np.random.choice(X.shape[1], size=num_features, replace=False)
    # Remove selected features
    X_augmented = np.delete(X, drop_indices, axis=1)
    return X_augmented

def save_X_and_Y(X, Y):
    np.save('X.npy', X)
    np.save('Y.npy', Y)







