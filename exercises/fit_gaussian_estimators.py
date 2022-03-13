import pandas as pd

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"
pio.renderers.default = "browser"  # todo: remove (and figure out how to show without)


def generate_samples(expected_value=10, variance=1, samples_count=1000):
    sd = np.sqrt(variance)
    return np.random.normal(expected_value, sd, samples_count)


def fit_single_univariate_sample(samples):
    samples = generate_samples()
    gaussian = UnivariateGaussian()
    return gaussian.fit(samples)


def question2(true_mu, true_var):
    df = pd.DataFrame([], columns=["Sample_Size", "Diff_mu", "Diff_var"])
    for sample_size in range(10, 1001, 10):
        samples = generate_samples(samples_count=sample_size)
        fitted_gaussian = fit_single_univariate_sample(samples)
        df = df.append({"Sample_Size": sample_size,
                        "Diff_mu": np.abs(fitted_gaussian.mu_ - true_mu),
                        "Diff_var": np.abs(fitted_gaussian.var_ - true_var)}, ignore_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Sample_Size"], y=df["Diff_mu"], mode='markers', name="diff mu"))
    fig.add_trace(go.Scatter(x=df["Sample_Size"], y=df["Diff_var"], mode='markers', name="diff var"))
    fig.update_xaxes(title_text="Sample Size")
    fig.update_yaxes(title_text="Diff")
    fig.update_layout(height=500)
    # todo add title to the graph
    fig.show()


def question3(fitted_gaussian, samples):
    pdf = fitted_gaussian.pdf(samples)
    df = pd.DataFrame([], columns=["Sample", "pdf"])
    df["Sample"] = np.array(samples)
    df["pdf"] = pdf

    df.sort_values(by="Sample")

    from scipy import stats
    xrange = np.arange(6, 14, 0.0025)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Sample"], y=df["pdf"], mode="markers"))
    fig.add_trace(go.Scatter(x=xrange, y=stats.norm(loc=10, scale=1).pdf(xrange), mode='lines',
                             line=dict(width=1.5, color="red"), name='Normal'))
    fig.update_xaxes(title_text="Sample Value")
    fig.update_yaxes(title_text="PDF Value")
    fig.update_layout(title="Sample to calculated Pdf values", height=500)  # todo find better title and
    # make it in the center

    fig.show()

    # todo the stats norm curve is probably to be deleted, though is is pretty :)
    # todo write "What are you expecting to see in the plot?" (from ex instructions)


def test_univariate_gaussian():
    true_mu, true_var = 10, 1

    # Question 1 - Draw samples and print fitted model
    samples = generate_samples()
    fitted_gaussian = fit_single_univariate_sample(samples)
    print(f"({fitted_gaussian.mu_}, {fitted_gaussian.var_})")

    # # Question 2 - Empirically showing sample mean is consistent
    question2(true_mu, true_var)

    # # Question 3 - Plotting Empirical PDF of fitted model
    question3(fitted_gaussian, samples)


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
