"""
Perform simple Parallel Tempering with two chains
"""
import numpy as np

LAM = 12


def likelihood(x):
    """
    Likelihood of x

    Parameters
    ----------
    x : 1d-array or 2d-array
        Array containing the horizontal and vertical component of one or several
        vectors.

    Returns
    -------
    likelihood : float or 1d-array
        Likelihood of vector x.
    """
    # Convert the argument into 2d-array in order to make this function capable of
    # computing likelihoods for multiple vectors
    x = np.atleast_2d(x)
    x1 = x[:, 0]
    x2 = x[:, 1]
    return np.exp(-(x1 ** 2 + x2 ** 2 + (x1 * x2) ** 2) - 2 * LAM * x1 * x2 / 2)


# Define parameters
iterations = int(10e3)
temperatures = [1, 10]  # temperatures of the chains
probability_of_swap = 0.5  # the probability to attempt a swap
iterations_swap = 100  # attempt a swap after this number of iterations
sigma = [0.1, 2]

# Initialize chains
n_chains = len(temperatures)
x = np.zeros(n_chains, 2)
probability = np.zeros(n_chains)
for chain in range(n_chains):
    # Proposal
    x[chain, :] = sigma[chain] * np.random.randn(2) + np.array([5, 5])
    # Compute probability of the x vector on each chain
    probability[chain] = likelihood(x[chain, :])


# Perform Parallel Tempering
# --------------------------
for iteration in range(iterations):
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
            x[0, :], x[1, :] = x[1, :], x[0, :]
            probability[0, :], probability[1, :] = probability[1, :], probability[0, :]

    # Perform MCMC
    for chain in range(n_chains):
        x_trial = x[chain, :] + sigma[chain] * np.random.randn(2)
        probability_trial = likelihood(x_trial)
        acceptance = min(
            1, (probability_trial / probability[chain]) ** (1 / temperatures[chain])
        )
        if acceptance > np.rand.rand():
            x[chain, :] = x_trial
            probability[chain, :] = probability_trial
