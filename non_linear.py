"""
Solve a non-linear inverse problem with MCMC
"""
import numpy as np
from scipy import stats


def forward(model):
    """
    Compute forward model
    """
    return model ** 2 + model


def prior(model, reference_model, error, sigma):
    """
    Compute the prior PDF as a Gaussian function
    """
    ss_prior = sum(((model - reference_model) / error) ** 2)
    # We need the square root of ss_prior, because the first argument of stats.norm.pdf
    # will be powered to 2 when computing the PDF (see docstring of stats.norm.pdf)
    ss_prior = np.sqrt(ss_prior)
    return stats.norm.pdf(ss_prior, loc=0, scale=sigma)


def likelihood(data, data_predicted, sigma):
    """
    Compute the likelihood PDF as a Gaussian function
    """
    ss = sum((data - data_predicted) ** 2)
    # We need the square root of ss, because the first argument of stats.norm.pdf
    # will be powered to 2 when computing the PDF (see docstring of stats.norm.pdf)
    ss = np.sqrt(ss)
    return stats.norm.pdf(ss, loc=0, scale=sigma)


# Synthetic model
# ---------------
# Generate a synthetic model, compute the data it generates from the forward and
# contaminate with noise
model_synth = np.array([2.5, 4])
error = 0.1
data = forward(model_synth) + error * np.random.randn(2)
