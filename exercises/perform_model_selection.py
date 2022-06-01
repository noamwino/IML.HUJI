from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def question_1(n_samples, noise):
    X, epsilon = np.linspace(-1.2, 2, n_samples), np.random.normal(0, noise, size=n_samples)
    y_without_noise = (X+3) * (X+2) * (X+1) * (X-1) * (X-2)
    y = y_without_noise + epsilon
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), train_proportion=2/3)

    train_X, test_X = train_X.to_numpy()[:, 0], test_X.to_numpy()[:, 0]
    train_y, test_y = train_y.to_numpy(), test_y.to_numpy()

    fig = go.Figure()
    fig.add_traces([
        go.Scatter(x=X, y=y_without_noise, mode='markers+lines', marker=dict(color='blue'), name='True Model'),
        go.Scatter(x=train_X, y=train_y, mode='markers', marker=dict(color='red'), name='Train data'),
        go.Scatter(x=test_X, y=test_y, mode='markers', marker=dict(color='green'), name='Test data'),
    ])
    fig.update_layout(title=f"Q1: Generated dataset of {n_samples} samples with & without noise (noise "
                            f"level of {noise})".title())
    fig.show()

    return train_X, train_y, test_X, test_y


def question_2(train_X, train_y, n_samples, noise):
    train_scores, validate_scores = np.zeros(11), np.zeros(11)
    for k in range(0, 11):
        train_scores[k], validate_scores[k] = cross_validate(PolynomialFitting(k), train_X, train_y,
                                                             mean_square_error, 5)

    fig = go.Figure()
    fig.add_traces([
        go.Scatter(x=np.arange(0, 12), y=train_scores, mode="markers+lines", name="train"),
        go.Scatter(x=np.arange(0, 12), y=validate_scores, mode="markers+lines", name="validation")
    ])
    fig.update_xaxes(title='polynomial degree', tick0=0, dtick=1)
    fig.update_yaxes(title='loss (mean square error)', tick0=0, dtick=10)
    fig.update_layout(title=f"Q2: 5-fold Cross-Validation for different polynomial degrees "
                            f"(n_samples={n_samples}, noise={noise})".title())
    fig.show()

    return validate_scores


def question_3(validation_scores, train_X, train_y, test_X, test_y, n_samples, noise):
    k = np.argmin(validation_scores)
    model = PolynomialFitting(int(k)).fit(train_X, train_y)
    print(f"Results when n_samples={n_samples} and noise={noise}:")
    print(f"Best k: {k}, test_error: {round(model.loss(test_X, test_y), 2)}. "
          f"Previously validation error: {validation_scores[k]}\n")


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    train_X, train_y, test_X, test_y = question_1(n_samples, noise)

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    validation_scores = question_2(train_X, train_y, n_samples, noise)

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    question_3(validation_scores, train_X, train_y, test_X, test_y, n_samples, noise)


def question_6(n_samples):
    X, y = datasets.load_diabetes(return_X_y=True)
    return X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]


def question_7(train_X, train_y, n_evaluations):
    lambdas = np.linspace(0.1, 4, n_evaluations)

    ridge_train_scores, ridge_validate_scores = np.zeros(len(lambdas)), np.zeros(len(lambdas))
    lasso_train_scores, lasso_validate_scores = np.zeros(len(lambdas)), np.zeros(len(lambdas))

    for i, lam in enumerate(lambdas):
        # ridge
        ridge_train_scores[i], ridge_validate_scores[i] = cross_validate(RidgeRegression(lam), train_X,
                                                                         train_y, mean_square_error, 5)
        # lasso
        lasso_train_scores[i], lasso_validate_scores[i] = cross_validate(Lasso(lam), train_X, train_y,
                                                                         mean_square_error, 5)

    fig = go.Figure()
    fig.add_traces([
        go.Scatter(x=lambdas, y=ridge_train_scores, mode="markers+lines", name="Ridge train"),
        go.Scatter(x=lambdas, y=ridge_validate_scores, mode="markers+lines", name="Ridge validation"),
        go.Scatter(x=lambdas, y=lasso_train_scores, mode="markers+lines", name="Lasso train"),
        go.Scatter(x=lambdas, y=lasso_validate_scores, mode="markers+lines", name="Lasso validation")
    ])
    fig.update_xaxes(title='Regularization parameter').update_yaxes(title='Loss (mean square error)')
    fig.update_layout(title=f"Q7: CV for different values of the regularization parameter for Ridge and "
                            f"Lasso regressions".title())
    fig.show()

    return lambdas, ridge_validate_scores, lasso_validate_scores


def question_8(lambdas, ridge_validate_scores, lasso_validate_scores, train_X, train_y, test_X, test_y):
    best_ridge_lambda = lambdas[np.argmin(ridge_validate_scores)]
    best_lasso_lambda = lambdas[np.argmin(lasso_validate_scores)]
    print(f"Regularization parameter value achieved the best validation error for Ridge: {best_ridge_lambda}")
    print(f"Regularization parameter value achieved the best validation error for Lasso: {best_lasso_lambda}")

    ridge = RidgeRegression(best_ridge_lambda)
    lasso = Lasso(best_ridge_lambda)
    ls_regressor = LinearRegression()

    ridge.fit(train_X, train_y)
    lasso.fit(train_X, train_y)
    ls_regressor.fit(train_X, train_y)

    print(f"Ridge error over the test set: {ridge.loss(test_X, test_y)}")
    print(f"Lasso error over the test set: {mean_square_error(lasso.predict(test_X), test_y)}")
    print(f"Least Square error over the test set: {ls_regressor.loss(test_X, test_y)}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    train_X, train_y, test_X, test_y = question_6(n_samples)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas, ridge_validate_scores, lasso_validate_scores = question_7(train_X, train_y, n_evaluations)

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    question_8(lambdas, ridge_validate_scores, lasso_validate_scores, train_X, train_y, test_X, test_y)


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()

