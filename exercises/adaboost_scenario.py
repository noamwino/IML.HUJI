import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def question_1(train_X, train_y, test_X, test_y, n_learners, noise):
    adaboost = AdaBoost(wl=lambda: DecisionStump(), iterations=n_learners).fit(train_X, train_y)

    train_errors, test_errors = np.zeros(n_learners), np.zeros(n_learners)
    train_errors[0], test_errors[0] = np.inf, np.inf  # so 0 learners will not count as a good model

    for t in range(1, n_learners):
        train_errors[t] = adaboost.partial_loss(train_X, train_y, t)
        test_errors[t] = adaboost.partial_loss(test_X, test_y, t)

    fig = go.Figure(
        [
            go.Scatter(x=np.arange(1, n_learners + 1), y=train_errors[1:], name="train errors"),
            go.Scatter(x=np.arange(1, n_learners + 1), y=test_errors[1:], name="test errors")
        ],
        layout=go.Layout(title=f"Q1 - Train and test errors as a function of the number of learners in the "
                               f"ensemble (noise={noise})".title(),
                         xaxis=dict(title=r"Number of weak learners", tick0=0, dtick=25),
                         yaxis=dict(title=r"Misclassification error")
                         )
    )
    fig.show()
    return adaboost, test_errors


def question_2(adaboost, test_X, test_y, lims, noise):
    T = [5, 50, 100, 250]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{t} Weak Learners}}$" for t in T],
                        horizontal_spacing=0.03, vertical_spacing=.03)

    for i, weak_models_count in enumerate(T):
        fig.add_traces(
            [
                decision_surface(lambda X: adaboost.partial_predict(X, weak_models_count), lims[0], lims[1],
                                 showscale=False),
                go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                           marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                       line=dict(color="black", width=1)))
            ],
            rows=(i // 2) + 1, cols=(i % 2) + 1
        )

    fig.update_layout(title=f"Q2 - Decision boundaries per number of week learners (noise={noise})".title(),
                      margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()


def question_3(adaboost, test_X, test_y, test_errors, lims, noise):
    optimal_learners_count = int(np.argmin(test_errors))
    fig = go.Figure(
        [
            decision_surface(lambda X: adaboost.partial_predict(X, optimal_learners_count), lims[0], lims[1],
                             showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                       marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                   line=dict(color="black", width=1)))
        ],
    )
    accuracy_score = accuracy(test_y, adaboost.partial_predict(test_X, optimal_learners_count))
    fig.update_layout(title=f"Q3 - Decision Boundaries of the Optimal Ensemble - {optimal_learners_count} "
                            f"Decision Stumps, Accuracy: {accuracy_score} (noise={noise})")
    fig.show()


def question_4(adaboost, train_X, train_y, lims, noise):
    D = adaboost.D_ / np.max(adaboost.D_) * (20 if noise == 0 else 10)
    fig = go.Figure(
        [
            decision_surface(lambda X: adaboost.partial_predict(X, adaboost.iterations_), lims[0], lims[1],
                             showscale=False),
            go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers', showlegend=False,
                       marker=dict(color=train_y, colorscale=class_colors(2), size=D))
        ],
    )
    fig.update_layout(title=f"Q4 - Decision boundaries of the full ensemble (noise={noise})".title())
    fig.show()


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost, test_errors = question_1(train_X, train_y, test_X, test_y, n_learners, noise)

    # Question 2: Plotting decision surfaces
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    question_2(adaboost, test_X, test_y, lims, noise)

    # Question 3: Decision surface of best performing ensemble
    question_3(adaboost, test_X, test_y, test_errors, lims, noise)

    # Question 4: Decision surface with weighted samples
    question_4(adaboost, train_X, train_y, lims, noise)


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
