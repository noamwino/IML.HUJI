from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import os
pio.templates.default = "simple_white"


HOUSE_PRICES_DATA = os.path.join("..", "datasets", "house_prices.csv")


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename).drop_duplicates()

    # the id col is irrelevant
    data.drop("id", axis=1, inplace=True)
    data.dropna(inplace=True)

    # drop invalid values
    data.drop(data[data["price"] < 0].index, inplace=True)       # price cannot be negative
    data.drop(data[data["sqft_lot15"] < 0].index, inplace=True)  # squared fit cannot be negative
    data.drop(data[data["date"] == '0'].index, inplace=True)     # dates cannot be '0'
    data.drop(data[(data["bedrooms"] == 0) & (data["bathrooms"] == 0)].index, inplace=True)  # houses should
    # have at least one room

    data.loc[data["bedrooms"] == 33, "bedrooms"] = 3  # fixing a type in the dataset

    # convert date to multiple columns
    as_datetime = pd.to_datetime(data["date"], format="%Y%m%dT000000")
    data["sold_year"] = as_datetime.dt.year
    data["sold_month"] = as_datetime.dt.month

    data.drop("date", axis=1, inplace=True)  # drop the original date

    # categorical features
    data = pd.get_dummies(data, columns=['zipcode', 'sold_year', 'sold_month'], drop_first=True)

    features = data.drop("price", axis=1)
    labels = data["price"]

    return features, labels


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature_name in X.columns:
        feature = X[feature_name]
        pearson_corr = np.cov(feature, y)[0][1] / (np.std(feature) * np.std(y))

        ordered_df = pd.concat((feature, y), axis=1).sort_values([feature_name, 'price'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ordered_df[feature_name], y=ordered_df["price"], mode='markers'))
        fig.update_xaxes(title=feature_name)
        fig.update_yaxes(title="price (response)")
        fig.update_layout(width=2000, height=800,
                          title=f"Pearson Correlation Between {feature_name} and prices (response) is "
                                f"{pearson_corr}")

        fig.write_image(os.path.join(output_path, feature_name + ".png"))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    features, labels = load_data(HOUSE_PRICES_DATA)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(features, labels, output_path="Q2Graphs")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(features, labels)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set

    df = pd.DataFrame({}, columns=["p", "mean_loss", "std_loss"])
    for p in range(10, 101):
        losses = np.array([])
        for _ in range(10):
            partial_train_X = train_X.sample(frac=p / 100)
            partial_train_y = train_y[partial_train_X.index]

            lin = LinearRegression()
            lin.fit(partial_train_X.to_numpy(), partial_train_y)
            loss = lin.loss(test_X.to_numpy(), test_y.to_numpy())
            losses = np.append(losses, loss)

        mean_loss = losses.mean()
        std_loss = losses.std()

        df = df.append({"p": p, "mean_loss": mean_loss, "std_loss": std_loss}, ignore_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["p"], y=df["mean_loss"], mode="markers+lines", name="Mean Loss"))
    fig.add_trace(go.Scatter(x=df["p"], y=df["mean_loss"]-2*df["std_loss"], mode="lines",
                             line=dict(color="lightgrey"), showlegend=False))
    fig.add_trace(go.Scatter(x=df["p"], y=df["mean_loss"]+2*df["std_loss"], mode="lines", fill='tonexty',
                             line=dict(color="lightgrey"), showlegend=False))
    fig.update_xaxes(title="percentage")
    fig.update_yaxes(title="Mean Loss")
    fig.update_layout(height=700, width=1500,
                      title="Q4: Mean Loss as a Function of the Percentage of the Train Taken")
    fig.show()
