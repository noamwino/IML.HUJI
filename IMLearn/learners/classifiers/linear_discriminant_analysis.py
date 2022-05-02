from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

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

        def cov_per_group(group):
            X, y = group[:, 0:-1], group[:, -1][0]  # all y's have the same label, enough to take the first
            class_index_in_mu = np.where(self.classes_ == y)[0][0]
            return (X - self.mu_[class_index_in_mu]).T @ (X - self.mu_[class_index_in_mu])

        # --- CALCULATIONS --- #
        self.classes_ = np.unique(y)
        self.pi_ = np.unique(y, return_counts=True)[1] / len(y)
        self.mu_ = np.array(list(map(lambda group: group[:, 0:-1].mean(axis=0), groups)))
        self.cov_ = np.array(list(map(cov_per_group, groups))).sum(axis=0) / (len(X) - len(self.classes_))
        self._cov_inv = inv(self.cov_)

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

        def likelihood_per_class(k):
            """ Returns an (n_samples,) array representing the likelihood of X to get the label k """

            return (np.exp(-.5 * np.einsum("bi,ij,bj->b", X-self.mu_[k], self._cov_inv, X-self.mu_[k])) /
                    np.sqrt((2*np.pi) ** X.shape[1] * det(self.cov_)) * self.pi_[k])[:, np.newaxis]

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
