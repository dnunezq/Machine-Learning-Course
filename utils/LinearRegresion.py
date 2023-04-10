from sklearn.linear_model import LinearRegression

class LinearRegressionModel:
    """
    A linear regression model that uses scikit-learn's LinearRegression class.
    
    Parameters:
    -----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    
    Attributes:
    -----------
    coef_ : array of shape (n_features,) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
    
    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0.0 if `fit_intercept=False`.
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=1, noise=10)
    >>> model = LinearRegressionModel()
    >>> model.fit(X, y)
    >>> model.predict([[0], [1]])
    array([-0.350...,  7.181...])
    """
    
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=self.fit_intercept)
    
    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        """
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
    
    def predict(self, X):
        """
        Predict using the linear regression model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        Returns:
        --------
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        return self.model.predict(X)