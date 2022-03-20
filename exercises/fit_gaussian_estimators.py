import pandas as pd

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def generate_univariate_samples(expected_value=10, variance=1, samples_count=1000):
    """ Generates samples from univariate gaussian distribution with the parameters given """
    sd = np.sqrt(variance)
    return np.random.normal(expected_value, sd, samples_count)


def generate_multivariate_samples(expected_value, cov, sample_count):
    """ Generates samples from multivariate gaussian distribution with the parameters given """
    return np.random.multivariate_normal(expected_value, cov, size=sample_count)


def fit_single_univariate_sample(samples):
    """ Fit a univariate gaussian model based on the samples given """
    gaussian = UnivariateGaussian()
    return gaussian.fit(samples)


def question2(true_mu, samples):
    """ Fitting a series of model with increasing sizes and plot as a function of the sample size """
    df = pd.DataFrame([], columns=["Sample_Size", "Diff_mu", "Diff_var"])
    for sample_size in range(10, 1001, 10):
        fitted_gaussian = fit_single_univariate_sample(samples[:sample_size])
        df = df.append({"Sample_Size": sample_size, "Diff_mu": np.abs(fitted_gaussian.mu_ - true_mu)},
                       ignore_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Sample_Size"], y=df["Diff_mu"], mode='markers'))

    fig.update_xaxes(title_text="Sample Size")
    fig.update_yaxes(title_text="Distance (|estimated - true|)")
    fig.update_layout(height=500, width=1000,
                      title="Q2: Absolute distance between estimated and true expectation value as a function"
                            " of sample size")
    fig.show()


def question3(fitted_gaussian, samples):
    """
    Computes the PDF of the previously drawn samples using the model fitted in question 1, and plots
    the empirical PDF function
    """
    pdf = fitted_gaussian.pdf(samples)
    df = pd.DataFrame([], columns=["Sample", "pdf"])
    df["Sample"] = np.array(samples)
    df["pdf"] = pdf

    df.sort_values(by="Sample")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Sample"], y=df["pdf"], mode="markers"))

    fig.update_xaxes(title_text="Sample Value", dtick=1)
    fig.update_yaxes(title_text="PDF Value")
    fig.update_layout(title="Q3: The empirical PDF function under the fitted model", width=1000, height=500)

    fig.show()


def question4(samples):
    """ Fits a multivariate Gaussian and print the estimated expectation and covariance matrix """
    gaussian = MultivariateGaussian()
    gaussian.fit(samples)
    print(gaussian.mu_)
    print(gaussian.cov_)


def question5(samples, cov):
    """
    Calculates the log-likelihood for models with different expectations, and plots a heatmap of the values
    ("as a function" of the different expectations)
    """
    f1_f3_pairs = np.array(np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))).transpose()\
        .reshape(-1, 2)
    df = pd.DataFrame({}, columns=["f1", "f3", "log-likelihood"])

    for f1, f3 in f1_f3_pairs:
        mu = np.array([f1, 0, f3, 0])
        log_likelihood = MultivariateGaussian.log_likelihood(mu, cov, samples)
        df = df.append({"f1": f1, "f3": f3, "log-likelihood": log_likelihood}, ignore_index=True)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=df["f3"], y=df["f1"], z=df["log-likelihood"], colorscale='Inferno'))
    fig.update_xaxes(title_text="f3", dtick=2)
    fig.update_yaxes(title_text="f1", dtick=2)
    fig.update_layout(height=700, width=700, title="Q5: Heatmap of log-likelihood values for f1 and f3")
    fig.show()

    return df


def question6(log_likelihood_data):
    """ Finds the model that achieve the maximum log likelihood and prints its parameters """
    max_value = log_likelihood_data["log-likelihood"].max()
    max_log_row = log_likelihood_data[log_likelihood_data["log-likelihood"] == max_value].iloc[0]

    print(f"Max Log-Likelihood is achieved when f1 = {max_log_row['f1']} and f3 = "
          f"{max_log_row['f3']}. The log-likelihood in this case: {max_value}")


def test_univariate_gaussian():
    true_mu, true_var = 10, 1

    # Question 1 - Draw samples and print fitted model
    samples = generate_univariate_samples()
    fitted_gaussian = fit_single_univariate_sample(samples)
    print(f"({fitted_gaussian.mu_}, {fitted_gaussian.var_})")

    # # Question 2 - Empirically showing sample mean is consistent
    question2(true_mu, samples)

    # # Question 3 - Plotting Empirical PDF of fitted model
    question3(fitted_gaussian, samples)


def test_multivariate_gaussian():
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[  1, 0.2, 0, 0.5],
                      [0.2,   2, 0,   0],
                      [  0,   0, 1,   0],
                      [0.5,   0, 0,   1]])
    samples = generate_multivariate_samples(mu, sigma, 1000)

    # Question 4 - Draw samples and print fitted model
    question4(samples)

    # # Question 5 - Likelihood evaluation
    log_likelihood_data = question5(samples, sigma)

    # # Question 6 - Maximum likelihood
    question6(log_likelihood_data)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
