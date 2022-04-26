from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # --- PREPARATIONS --- #
        # Creating a list in which each item is an array of samples from X sharing the same label
        Xy = np.c_[X, y.T]
        Xy = Xy[Xy[:, -1].argsort()]
        groups = np.split(Xy, np.unique(Xy[:, -1], return_index=True)[1][1:])

        # --- CALCULATIONS --- #
        self.classes_ = np.unique(y)  # same as LDA
        self.pi_ = np.unique(y, return_counts=True)[1] / len(y)  # same as LDA
        self.mu_ = np.array(list(map(lambda group: group[:, 0:-1].mean(axis=0), groups)))  # same as LDA
        self.vars_ = np.array(list(map(lambda group: group[:, 0:-1].var(axis=0, ddof=1), groups)))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.likelihood(X).argmax(axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        m, d = X.shape

        def likelihood_per_class(k):
            """ Returns an (n_samples,) array representing the likelihood of X to get the label k """
            mu, pi, sigma = self.mu_[k], self.pi_[k], np.diag(self.vars_[k])
            det_sigma, inv_sigma = np.linalg.det(sigma), np.linalg.inv(sigma)
            return (np.exp(-.5 * np.einsum("bi,ij,bj->b", X-mu, inv_sigma, X-mu)) /
                    np.sqrt((2*np.pi) ** d * det_sigma) * pi)[:, np.newaxis]

        return np.concatenate(list(map(likelihood_per_class, self.classes_)), axis=1)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
