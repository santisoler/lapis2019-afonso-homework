"""
Perform a simple Multiple Try Metropolis algorithm
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


# Define parameters like number of iterations, number of trials, sigma for the T(x, y)
# distribution
iterations = int(10e3)
n_trials = 20
sigma = 3

# Run MTM
# -------
data = []
x = np.array([0, 0])
for i in range(iterations):
    # Get y_trials around x with gaussian probability
    y_trials = x + sigma * np.random.randn(n_trials, 2)
    # Compute probability of each element of y_trials
    probabilities_y = likelihood(y_trials)
    # Select y among the y_trials with probability of pi(y_trials)
    # I have to normalize the probabilities in order to sum one
    index = np.random.choice(n_trials, p=probabilities_y / probabilities_y.sum())
    y = y_trials[index]

    # Generate the reference set (assign the current x to the last element of the set)
    reference_set = y + sigma * np.random.randn(n_trials, 2)
    reference_set[-1] = x
    # Compute the generalized M-H ratio
    rg = min(1, probabilities_y.sum() / likelihood(reference_set).sum())
    # Lets accept y with probability rg
    if rg > np.random.rand():
        data.append(y)
        x = y
data = np.array(data)


# Plot results and target PDF
x1 = np.linspace(-4, 4, 101)
x1, x2 = np.meshgrid(x1, x1)
target = likelihood(np.hstack((x1[:, np.newaxis], x2[:, np.newaxis])))
target = target.reshape(x1.shape)

plt.contour(x1, x2, target)
plt.scatter(data[:, 0], data[:, 1], s=1, label="Accepted points by MTM")
plt.axes().set_aspect("equal")
plt.grid()
plt.legend()
plt.show()
