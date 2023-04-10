from sklearn.neighbors import KNeighborsClassifier

class KNNModel:
    """
    A K-Nearest Neighbors classifier that uses scikit-learn's KNeighborsClassifier class.
    
    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of neighbors to use in the classification.
        
    weights : str or callable, default='uniform'
        Weight function used in prediction. Possible values:
        - 'uniform': uniform weights (all points in each neighborhood are weighted equally).
        - 'distance': weight points by the inverse of their distance. 
        
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors.
        
    Attributes:
    -----------
    classes_ : array of shape (n_classes,)
        The class labels (single output problem).
        
    Examples:
    ---------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> model = KNNModel(n_neighbors=3)
    >>> model.fit(X_train, y_train)
    >>> model.predict(X_test[:3])
    array([1, 0, 1])
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, algorithm=self.algorithm)
    
    def fit(self, X, y):
        """
        Fit the KNN model according to the given training data (X, y).
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels).
        """
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
    
    def predict(self, X):
        """
        Perform classification on samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns:
        --------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        return self.model.predict(X)