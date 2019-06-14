"""
Perform simple Parallel Tempering with two chains
"""
import numpy as np
import matplotlib.pyplot as plt

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
x = np.zeros((n_chains, 2))
probability = np.zeros(n_chains)
for chain in range(n_chains):
    # Proposal
    x[chain, :] = sigma[chain] * np.random.randn(2) + np.array([5, 5])
    # Compute probability of the x vector on each chain
    probability[chain] = likelihood(x[chain, :])
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
            x[0, :], x[1, :] = x[1, :], x[0, :]
            probability[0], probability[1] = probability[1], probability[0]

    # Perform MCMC
    for chain in range(n_chains):
        x_trial = x[chain, :] + sigma[chain] * np.random.randn(2)
        probability_trial = likelihood(x_trial)
        acceptance = min(
            1, (probability_trial / probability[chain]) ** (1 / temperatures[chain])
        )
        if acceptance > np.random.rand():
            x[chain, :] = x_trial
            probability[chain] = probability_trial
        # Add sampled points to array
        sampled_points[chain, i, :] = x[chain, :]


# Plot results and target PDF
x1 = np.linspace(-4, 4, 101)
x1, x2 = np.meshgrid(x1, x1)
target = likelihood(np.hstack((x1[:, np.newaxis], x2[:, np.newaxis])))
target = target.reshape(x1.shape)

fig, axes = plt.subplots(nrows=1, ncols=n_chains)
colors = ["C{}".format(i) for i in range(n_chains)]
for chain in range(n_chains):
    ax = axes[chain]
    ax.contour(x1, x2, target)
    ax.scatter(
        sampled_points[chain, :, 0],
        sampled_points[chain, :, 1],
        s=4,
        alpha=0.2,
        color=colors[chain],
    )
    ax.set_aspect("equal")
    ax.grid()
    ax.set_title("T = {}".format(temperatures[chain]))
plt.tight_layout()
plt.show()
