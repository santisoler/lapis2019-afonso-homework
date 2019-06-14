"""
Perform simple Parallel Tempering with multiple chains
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
temperatures = np.arange(1, 11)  # temperatures of the chains
probability_of_swap = 0.5  # the probability to attempt a swap
iterations_swap = 10  # attempt a swap after this number of iterations
sigma = temperatures / 5

# Initialize chains
n_chains = len(temperatures)
x = np.zeros((n_chains, 2))
probability = np.zeros(n_chains)
sampled_points = np.empty((n_chains, iterations, 2))
for chain in range(n_chains):
    # Proposal
    x[chain, :] = np.array([5, 5]) + sigma[chain] * np.random.randn(2)
    # Compute probability of the x vector on each chain
    probability[chain] = likelihood(x[chain, :])


# Perform Parallel Tempering
# --------------------------
for iteration in range(iterations):
    # Attempt swap every iteration_swap iterations
    if iteration % iterations_swap == 0:
        if probability_of_swap > np.random.rand():
            # Randomly choose two chains to attempt to swap
            i, j = np.random.choice(n_chains, 2)
            # Compute alpha between this pair of chains
            alpha = min(
                1,
                (probability[j] / probability[i]) ** (1 / temperatures[i])
                * (probability[i] / probability[j]) ** (1 / temperatures[j]),
            )
            # Decide if we should swap chains i and j
            if alpha > np.random.rand():
                x[i, :], x[j, :] = x[j, :], x[i, :]
                probability[i], probability[j] = probability[j], probability[i]

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
        sampled_points[chain, iteration, :] = x[chain, :]


# Plot results and target PDF
x1 = np.linspace(-4, 4, 101)
x1, x2 = np.meshgrid(x1, x1)
target = likelihood(np.hstack((x1[:, np.newaxis], x2[:, np.newaxis])))
target = target.reshape(x1.shape)

plt.contour(x1, x2, target)
for chain in range(n_chains)[::-1]:
    plt.scatter(
        sampled_points[chain, :, 0],
        sampled_points[chain, :, 1],
        s=4,
        alpha=0.2,
        label="T={}".format(temperatures[chain]),
    )
plt.axes().set_aspect("equal")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
