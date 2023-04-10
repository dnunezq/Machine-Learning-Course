from sklearn.svm import SVC

class SVMModel:
    """
    A support vector machine (SVM) classifier that uses scikit-learn's SVC class.
    
    Parameters:
    -----------
    kernel : str, default='rbf' (other options: {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'})
        Specifies the kernel type to be used in the algorithm. 
        
    C : float, default=1.0
        Penalty parameter of the error term. 
        
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. 
        
    Attributes:
    -----------
    support_vectors_ : array-like of shape (n_SV, n_features)
        The support vectors.
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> model = SVMModel()
    >>> model.fit(X_train, y_train)
    >>> model.predict(X_test[:3])
    array([1, 1, 1])
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
    
    def fit(self, X, y):
        """
        Fit the SVM model according to the given training data (X, y).
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels).
        """
        self.model.fit(X, y)
        self.support_vectors_ = self.model.support_vectors_
    
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