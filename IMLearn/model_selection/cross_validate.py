from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    Xy = np.c_[X, y.T]
    groups_Xy = np.array_split(Xy, cv)
    train_scores, validation_scores = np.zeros(cv), np.zeros(cv)

    for k in range(cv):
        fold = groups_Xy[k]                                       # the k'th group
        train = np.concatenate(groups_Xy[0:k] + groups_Xy[k+1:])  # All except from the k'th group
        train_X, train_y = train[:, :-1], train[:, -1]
        validation_X, validation_y = fold[:, :-1], fold[:, -1]

        if len(train_X.shape) == 2 and train_X.shape[1] == 1:
            train_X, validation_X = train_X.squeeze(), validation_X.squeeze()

        estimator.fit(train_X, train_y)
        train_scores[k] = scoring(train_y, estimator.predict(train_X))
        validation_scores[k] = scoring(validation_y, estimator.predict(validation_X))

    return train_scores.mean(), validation_scores.mean()

