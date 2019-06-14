"""
Solve a non-linear inverse problem with MCMC
"""
import numpy as np
import matplotlib.pyplot as plt
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


# ---------------
# Inverse problem
# ---------------
# Define parameters
iterations = int(10e3)
temperatures = [1, 10]  # temperatures of the chains
probability_of_swap = 0.5  # the probability to attempt a swap
sigma_mcmc = [0.1, 2]  # standard deviation for the MCMC steps for each chain
sigma_prior = 10  # standard deviation for the prior (we choose it to be big)
sigma_likelihood = 1  # standard deviation for the likelihood (we choose it small)
reference_model = [5, 5]  # reference model used on the prior

# Initialize chains
n_chains = len(temperatures)
models = np.zeros((n_chains, 2))
probability = np.zeros(n_chains)
for chain in range(n_chains):
    # Proposal
    models[chain, :] = np.array([5, 5]) + sigma_mcmc[chain] * np.random.randn(2)
    # Compute probability of the proposed models on each chain
    probability[chain] = likelihood(
        data, forward(models[chain]), sigma_likelihood
    ) * prior(models[chain], reference_model, error, sigma_prior)
sampled_points = np.empty((n_chains, iterations, 2))

# Perform Parallel Tempering
# --------------------------
for i in range(iterations):
    # Decide whether to swap chains or not
    if probability_of_swap > np.random.rand():
        # Compute alpha
        alpha = min(
            1,
            (probability[1] / probability[0]) ** (1 / temperatures[0])
            * (probability[0] / probability[1]) ** (1 / temperatures[1]),
        )
        # Decide if we should swap
        if alpha > np.random.rand():
            models[0, :], models[1, :] = models[1, :], models[0, :]
            probability[0], probability[1] = probability[1], probability[0]

    # Perform MCMC
    for chain in range(n_chains):
        model_trial = models[chain, :] + sigma_mcmc[chain] * np.random.randn(2)
        # Lets compute the posterior PDF
        probability_trial = likelihood(
            data, forward(model_trial), sigma_likelihood
        ) * prior(model_trial, reference_model, error, sigma_prior)
        # Compute the acceptance probability
        acceptance = min(
            1, (probability_trial / probability[chain]) ** (1 / temperatures[chain])
        )
        if acceptance > np.random.rand():
            models[chain, :] = model_trial
            probability[chain] = probability_trial
        # Add sampled points to array
        sampled_points[chain, i, :] = models[chain, :]


# ------------
# Plot results
# ------------
for chain in range(n_chains)[::-1]:
    plt.scatter(
        sampled_points[chain, :, 0],
        sampled_points[chain, :, 1],
        s=4,
        alpha=0.4,
        label="T={}".format(temperatures[chain]),
    )
plt.axes().set_aspect("equal")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


# Plot histograms
# ---------------
colors = ["C0", "C1"]
fig, axes = plt.subplots(nrows=n_chains, ncols=2)
for chain in range(n_chains):
    ax_row = axes[chain, :]
    for coordinate in range(2):
        ax_row[coordinate].hist(
            sampled_points[chain, :, coordinate],
            color=colors[chain],
            label="T={}".format(temperatures[chain]),
        )
        ax_row[coordinate].legend()
        ax_row[coordinate].set_title("Model element {}".format(coordinate + 1))
plt.show()


# Trace plots
# -----------
colors = ["C0", "C1"]
fig, axes = plt.subplots(nrows=2 * n_chains, ncols=1, sharex=True)
i = 0
for chain in range(n_chains):
    for coordinate in range(2):
        axes[i].plot(
            sampled_points[chain, :, coordinate],
            color=colors[chain],
            label="Model element={}, T={}".format(coordinate + 1, temperatures[chain]),
        )
        axes[i].legend()
        i += 1
plt.show()
