from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import os


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for dataset_name, file_name in [("Linearly Separable", "linearly_separable.npy"),
                                    ("Linearly Inseparable", "linearly_inseparable.npy")]:
        X, y = load_dataset(os.path.join("..", "datasets", file_name))
        losses = []

        def record_losses(fit: Perceptron, xi: np.ndarray, yi: int):
            losses.append(fit.loss(X, y))

        Perceptron(callback=record_losses).fit(X, y)

        fig = px.line(x=np.arange(start=1, stop=len(losses) + 1), y=np.array(losses), title=dataset_name,
                      labels={"x": "Number of Iterations", "y": "Mis-classification Loss (normalized)"},
                      height=700, width=1500)
        fig.update_yaxes(range=[0, 1], tick0=0, dtick=0.1)  # the losses are normalized
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def scatter_points(X: np.ndarray, y: np.ndarray, symbols: np.ndarray, predictions: np.ndarray):
    """
    returns a scatter plot of the features of X, colored by the predictions and shaped by the true labels y.
    """
    return go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                      marker=dict(color=predictions, symbol=symbols[y], line=dict(width=1)))


def center_of_group(mu: np.ndarray):
    """
    returns a scatter plot with one dot representing the center of a group with expectation value mu.
    """
    return go.Scatter(x=[mu[0]], y=[mu[1]], mode="markers",  marker=dict(color="black", symbol="x", size=15),
                      showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    from IMLearn.metrics import accuracy
    symbols = np.array(["circle", "x", "diamond"])

    for file_name in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(os.path.join("..", "datasets", file_name))

        # Fit models and predict over training set
        lda = LDA()
        lda_predictions = lda.fit(X, y).predict(X)
        lda_accuracy = accuracy(y, lda_predictions)

        naive = GaussianNaiveBayes()
        naive_predictions = naive.fit(X, y).predict(X)
        naive_accuracy = accuracy(y, naive_predictions)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA
        # predictions on the right. Plot title should specify dataset used and subplot titles should
        # specify algorithm and accuracy Create subplots

        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.01, vertical_spacing=.03,
                            subplot_titles=[f"Naive Bayes Classifier Over {file_name}, Accuracy: {naive_accuracy}",
                                            f"LDA Classifier Over {file_name}, Accuracy: {lda_accuracy}"])

        # Add traces for data-points setting symbols and colors
        fig.add_trace(scatter_points(X, y, symbols, lda_predictions), row=1, col=2)
        fig.add_trace(scatter_points(X, y, symbols, naive_predictions), row=1, col=1)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for k in np.unique(y):  # k is an optional class name
            # lda
            fig.add_trace(center_of_group(lda.mu_[k]), row=1, col=2)
            fig.add_trace(get_ellipse(lda.mu_[k], lda.cov_), row=1, col=2)

            # naive gaussian
            fig.add_trace(center_of_group(naive.mu_[k]), row=1, col=1)
            fig.add_trace(get_ellipse(naive.mu_[k], np.diag(naive.vars_[k])), row=1, col=1)

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
