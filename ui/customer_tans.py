# custom_transformers.py
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features, all_features):
        self.selected_features = selected_features
        self.all_features = all_features
        self.indices_ = None

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.indices_ = [X.columns.get_loc(f) for f in self.selected_features]
        else:
            self.indices_ = [self.all_features.index(f) for f in self.selected_features]
        return self

    def transform(self, X):
        if hasattr(X, 'iloc'):
            return X[self.selected_features].values
        else:
            return X[:, self.indices_]