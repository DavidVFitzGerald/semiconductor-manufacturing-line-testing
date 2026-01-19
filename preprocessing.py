import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class HighNaNColumnDropper(BaseEstimator, TransformerMixin):
    """Drop columns with fraction of NaNs greater than a threshold."""

    def __init__(self, nan_threshold=0.5):
        self.nan_threshold = nan_threshold
        self.keep_mask_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_samples = X.shape[0]
        nan_frac = np.isnan(X).sum(axis=0) / float(n_samples)
        self.keep_mask_ = nan_frac <= self.nan_threshold
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.keep_mask_]


class ConstantColumnDropper(BaseEstimator, TransformerMixin):
    """Drop columns that are constant (<= 1 unique non-NaN value)."""

    def __init__(self):
        self.keep_mask_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        unique_counts = [len(np.unique(col[~np.isnan(col)])) for col in X.T]
        self.keep_mask_ = np.array([uc > 1 for uc in unique_counts], dtype=bool)
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.keep_mask_]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Remove highly correlated features (keep higher-variance feature)."""

    def __init__(self, corr_threshold=0.9):
        self.corr_threshold = corr_threshold
        self.keep_mask_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        # Expect no NaNs (imputer should run before this)
        corr = np.corrcoef(X, rowvar=False)
        corr = np.abs(corr)
        n_features = corr.shape[0]
        to_drop = set()
        for i in range(n_features):
            if i in to_drop:
                continue
            for j in range(i + 1, n_features):
                if j in to_drop:
                    continue
                if corr[i, j] >= self.corr_threshold:
                    var_i = np.nanvar(X[:, i])
                    var_j = np.nanvar(X[:, j])
                    if var_i >= var_j:
                        to_drop.add(j)
                    else:
                        to_drop.add(i)
        keep = [i for i in range(n_features) if i not in to_drop]
        mask = np.zeros(n_features, dtype=bool)
        mask[keep] = True
        self.keep_mask_ = mask
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.keep_mask_]
