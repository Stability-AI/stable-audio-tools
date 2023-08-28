# PCA implementation modified from https://github.com/gngdb/pytorch-pca/blob/main/pca.py

import torch
import torch.nn as nn


def svd_flip(u, v):
    """Flip the signs of columns of u and rows of v to ensure deterministic output"""
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    signs = torch.sign(u[max_abs_cols, torch.arange(u.shape[1])])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v

class PCA(nn.Module):
    """PyTorch implementation of PCA using SVD"""

    def __init__(self, n_components=32):
        super().__init__()
        self.n_components = n_components

        self.register_buffer("mean_", torch.zeros(1, self.n_components))
        self.register_buffer("components_", torch.zeros(self.n_components, self.n_components))
        self.register_buffer("explained_variance_", torch.zeros(self.n_components))
        self.register_buffer("explained_variance_ratio_", torch.zeros(self.n_components))
        self.register_buffer("has_fit_", torch.tensor(False))
        
    @torch.no_grad()
    def fit(self, X):
        """Fit the PCA model to the input data X"""
        n, d = X.size()

        # Set the number of components to retain if specified
        if self.n_components is not None:
            d = min(self.n_components, d)

        # Compute the mean of the input data
        self.mean_ = X.mean(0, keepdim=True)

        # Center the input data by subtracting the mean
        Z = X - self.mean_

        # Compute the SVD of the centered data
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)

        # Flip the signs of U and Vh to ensure deterministic output
        U, Vh = svd_flip(U, Vh)

        # Store the first d principal components as the "components" attribute
        self.components_ = Vh[:d]

        # Compute the explained variance of each principal component
        self.explained_variance_ = torch.square(S) / (n - 1)

        # Compute the proportion of the total variance explained by each principal component
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()

        self.has_fit_ = torch.tensor(True)

        return self

    def forward(self, X):
        """Apply the PCA transformation to input data X"""
        return self.transform(X)

    def transform(self, X):
        """Transform input data X to the principal component space"""
        assert self.has_fit_.item() , "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        """Fit the PCA model to input data X and transform it to the principal component space"""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        """Transform principal component space data Y back to the original space"""
        assert self.has_fit_.item() , "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_