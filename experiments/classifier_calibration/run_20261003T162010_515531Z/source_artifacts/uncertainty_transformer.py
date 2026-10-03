# IMPROVED UNCERTAINTY TRANSFORMER WITH PROPER NaN HANDLING
# This file contains the fixed version with better NaN handling options

"""
Uncertainty Transformer for binary classification with improved NaN handling.

This module provides a scikit-learn compatible transformer that computes
class-conditional statistics and uncertainty-weighted features for binary classification.
"""

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import warnings
from sklearn.base import BaseEstimator, TransformerMixin

from uncertainty_utils import (
    discretise,
    get_distribution,
    js_divergence,
    shannon_entropy,
)


@dataclass
class _FeatStats:
    """Per-feature statistics for uncertainty transformation."""
    mu: Dict[Hashable, float]
    std: Dict[Hashable, float]
    entropy: Dict[Hashable, float]
    js: float


class UncertaintyTransformer(BaseEstimator, TransformerMixin):
    """
    Uncertainty-weighted feature transformer for binary classification with improved NaN handling.

    This transformer computes class-conditional statistics (mean, std, entropy)
    and Jensen-Shannon divergence for each feature. During transformation, each
    feature value is mapped to: z * (entropy / (js + eps)), where z is the
    z-score relative to the closest class mean (in absolute z-score).

    The transformer is label-aware and must be fitted inside each CV fold to
    prevent data leakage.

    Parameters
    ----------
    feature_names : iterable of str, optional
        Names of features to transform. If None, inferred from X in fit().
    class_labels : tuple of (label1, label2), optional
        Binary class labels. If None, inferred from y in fit().
    n_bins : int, default=20
        Number of bins for discretization of continuous features.
    eps : float, default=1e-12
        Small epsilon to prevent division by zero.

    Notes
    -----
    This transformer requires clean data with NO missing values. If NaN values
    are present in the input data, the transform will raise an error.
    For clinical data, eliminate all rows with missing values before transformation
    to maintain data integrity (no imputation).

    Attributes
    ----------
    classes_ : tuple
        The two class labels used for transformation.
    feature_names_in_ : list of str
        Names of features seen during fit.
    _feat_stats_ : dict
        Per-feature statistics computed during fit.

    Examples
    --------
    >>> from uncertainty_transformer import UncertaintyTransformer
    >>> # Ensure data has no missing values before transformation
    >>> X_train_clean = X_train.dropna()
    >>> y_train_clean = y_train[X_train.notna().all(axis=1)]
    >>> transformer = UncertaintyTransformer(feature_names=['age', 'height'])
    >>> transformer.fit(X_train_clean, y_train_clean)
    >>> X_transformed = transformer.transform(X_test_clean)
    """

    def __init__(
        self,
        *,
        feature_names: Optional[Iterable[str]] = None,
        class_labels: Optional[Tuple[Hashable, Hashable]] = None,
        n_bins: int = 20,
        eps: float = 1e-12,
    ) -> None:
        self.feature_names = feature_names
        self.class_labels = class_labels
        self.n_bins = n_bins
        self.eps = eps

    def fit(self, X: Any, y: Any) -> "UncertaintyTransformer":
        """
        Fit the transformer by computing class-conditional statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Binary class labels.

        Returns
        -------
        self : UncertaintyTransformer
            Returns self for method chaining.
        """
        X_df = self._to_frame(X)
        y_arr = np.asarray(y)

        if y_arr.ndim != 1:
            raise ValueError("y must be 1D.")

        classes = np.unique(y_arr)
        present = set(classes)

        if self.class_labels is not None:
            c1, c2 = self.class_labels
            # Guard: ensure both provided class labels are present in y
            if c1 not in present or c2 not in present:
                raise ValueError(
                    f"Provided class_labels {self.class_labels} not both present in y: {present}. "
                    "This would result in 0 features being computed, causing downstream failures."
                )
        else:
            if len(classes) != 2:
                raise ValueError(
                    f"UncertaintyTransformer requires exactly 2 classes; got {classes}. "
                    "Pass class_labels=(c1, c2) if you want to select two classes."
                )
            c1, c2 = classes[0], classes[1]

        self.classes_ = (c1, c2)

        feats = list(self.feature_names) if self.feature_names is not None else list(X_df.columns)
        self.feature_names_in_ = list(feats)

        # Fail fast if any expected columns are missing
        missing = [c for c in self.feature_names_in_ if c not in X_df.columns]
        if missing:
            raise KeyError(f"UncertaintyTransformer: missing columns in X: {missing}")

        feat_stats: Dict[str, _FeatStats] = {}
        dropped_features = []  # Track dropped features for warning

        for feat in self.feature_names_in_:
            s = X_df[feat]
            # discretise uses self.n_bins parameter
            disc = discretise(s, n_bins=self.n_bins)

            p1 = get_distribution(disc[y_arr == c1])
            p2 = get_distribution(disc[y_arr == c2])
            if len(p1) == 0 or len(p2) == 0:
                dropped_features.append(feat)
                continue

            vals1 = s[y_arr == c1].dropna()
            vals2 = s[y_arr == c2].dropna()
            if len(vals1) == 0 or len(vals2) == 0:
                dropped_features.append(feat)
                continue

            # All computations use self.eps for consistency:
            # - std: add self.eps to prevent division by zero
            # - entropy: pass self.eps to shannon_entropy
            # - js: pass self.eps to js_divergence and use as minimum value
            feat_stats[feat] = _FeatStats(
                mu={c1: float(vals1.mean()), c2: float(vals2.mean())},
                std={
                    c1: float(vals1.std(ddof=0) + self.eps),
                    c2: float(vals2.std(ddof=0) + self.eps),
                },
                entropy={
                    c1: shannon_entropy(p1, eps=self.eps),
                    c2: shannon_entropy(p2, eps=self.eps),
                },
                js=max(self.eps, js_divergence(p1, p2, eps=self.eps)),
            )

        # Warn about dropped features
        if dropped_features:
            warnings.warn(
                f"UncertaintyTransformer dropped {len(dropped_features)} features due to "
                f"insufficient data in one or both classes: {dropped_features[:5]}"
                f"{'...' if len(dropped_features) > 5 else ''}",
                UserWarning
            )

        self._feat_stats_ = feat_stats
        self.feature_names_in_ = [feat for feat in feats if feat in feat_stats]

        return self

    def transform(self, X: Any) -> np.ndarray:
        """
        Transform features using fitted uncertainty statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed features.

        Raises
        ------
        ValueError
            If NaN values are detected in the transformation output.
            All input data must have missing values eliminated before transformation.
        """
        if not hasattr(self, "_feat_stats_"):
            raise RuntimeError("UncertaintyTransformer is not fitted yet.")

        X_df = self._to_frame(X)
        feats = self.feature_names_in_
        c1, c2 = self.classes_

        # build in consistent feature order
        out = np.zeros((len(X_df), len(feats)), dtype=float)

        for j, feat in enumerate(feats):
            fs = self._feat_stats_.get(feat)
            if fs is None or feat not in X_df.columns:
                continue

            v = X_df[feat].astype(float)

            # z-scores to each class
            z1 = (v - fs.mu[c1]) / fs.std[c1]
            z2 = (v - fs.mu[c2]) / fs.std[c2]

            # Replace infinities in z-scores with 0 (can occur if std is very small)
            # This prevents infinities from propagating to the final output
            z1 = z1.replace([np.inf, -np.inf], 0.0)
            z2 = z2.replace([np.inf, -np.inf], 0.0)

            # choose closest class by absolute z; ties go to class 1
            use_c1 = (np.abs(z1) <= np.abs(z2)).to_numpy()
            z = np.where(use_c1, z1.to_numpy(), z2.to_numpy())
            cls_entropy = np.where(use_c1, fs.entropy[c1], fs.entropy[c2])

            # Use self.eps in denominator to prevent division by zero
            out[:, j] = z * (cls_entropy / (fs.js + self.eps))

        # Check for NaN values - raise error if any are found
        nan_mask = np.isnan(out)
        nan_count = np.sum(nan_mask)

        if nan_count > 0:
            # Count NaNs per feature for informative error message
            nan_per_feat = np.sum(nan_mask, axis=0)
            nan_features = [feats[i] for i in range(len(feats)) if nan_per_feat[i] > 0]
            raise ValueError(
                f"NaN values detected in transform output. "
                f"Found {nan_count} NaN values across {len(nan_features)} features: "
                f"{nan_features[:5]}{'...' if len(nan_features) > 5 else ''}. "
                f"\n\nFor clinical data, eliminate all rows with missing values "
                f"BEFORE transformation:\n"
                f"  X_clean = X[~X.isna().any(axis=1)]  # Drop rows with any NaN\n"
                f"  y_clean = y[~X.isna().any(axis=1)]  # Keep corresponding labels\n\n"
                f"This transformer does not perform imputation to maintain clinical data integrity."
            )

        return out

    def _to_frame(self, X: Any) -> pd.DataFrame:
        """Convert input to pandas DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError("X numpy array must be 2D.")
            if self.feature_names is None and not hasattr(self, "feature_names_in_"):
                # fallback names
                cols = [f"x{i}" for i in range(X.shape[1])]
            else:
                cols = self.feature_names_in_ if hasattr(self, "feature_names_in_") else self.feature_names
            return pd.DataFrame(X, columns=list(cols))
        # try pandas conversion
        return pd.DataFrame(X)
