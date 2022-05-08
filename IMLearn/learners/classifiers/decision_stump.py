from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics.loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples, n_features = X.shape
        self.sign_ = 1  # todo check sign (+1 or -1?)

        # first feature
        threshold_plus1, error_plus1 = self._find_threshold(X[:, 0], y, 1)
        threshold_minus1, error_minus1 = self._find_threshold(X[:, 0], y, -1)

        min_error = min(error_plus1, error_minus1)
        self.j_ = 0
        self.sign_ = 1 if error_plus1 < error_minus1 else -1
        self.threshold_ = threshold_plus1 if error_plus1 < error_minus1 else threshold_minus1

        for feature_index in range(1, n_features):
            for sign in [-1, 1]:
                threshold, error = self._find_threshold(X[:, feature_index], y, sign)  # todo check sign (+1 or -1?)
                print(f"Feature {feature_index}, got the error {error} with threshold {threshold} and sign {sign}")
                if error < min_error:
                    print(f"Updating")
                    min_error = error
                    self.j_, self.threshold_, self.sign_ = feature_index, threshold, sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] < self.threshold_, -1 * self.sign_, self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        optimal_threshold, optimal_error = None, None

        for optional_threshold in values:
            print("Checking value", optional_threshold)
            prediction = np.where(values < optional_threshold, -1 * sign, sign)
            error = misclassification_error(labels, prediction)

            if not optimal_error or error < optimal_error:
                optimal_threshold = optional_threshold
                optimal_error = error

        return optimal_threshold, optimal_error

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
        return misclassification_error(self.predict(X), y)
