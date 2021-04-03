#%%
import numpy as np
from activations import SIGMOID
from sklearn.datasets import make_classification
import seaborn as sns

sigmoid = SIGMOID()

#%%
X, y = make_classification(n_features=2, n_informative=1,
    n_redundant=0, n_clusters_per_class=1)
X: np.ndarray
len(X),len(X[0]),len(y)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)

#%%
n, p = X.shape
biases = np.ones((n, 1))
X = np.append(X, biases, axis=1)
X

#%% [markdown]
# Instatiate parameter θ as a column vector.

#%%
θ = np.zeros((p + 1, 1))
θ

#%%
X.shape, θ.shape
np.array([[1,2]]) @ \
np.array([[3],[4]])

#%% Compute cost
def compute_cost(X, y, θ):
    m = len(y)
    h = sigmoid.func(X @ θ)
    epsilon = 1e-5
    return (1/m)*(  ( -y).T @ np.log(  h + epsilon) \
        -
                    (1-y).T @ np.log(1-h + epsilon))
compute_cost(X, y, θ)

#%%
def gradient_descent(X, y, θ, lr=0.03, iters=3000):
    m = len(y)
    cost_history = np.zeros((iters, 1))

    for i in range(iters):
        θ = θ - (lr/m) * (X.T @ (sigmoid.func(X @ θ) - y))
        cost_history[i] = compute_cost(X, y, θ).sum().item()
    return (cost_history, θ)

(hist, θ_opt) = gradient_descent(X, y, θ)
hist.shape, θ_opt.shape

#%%
sns.lineplot(x=range(len(hist)), y=hist[:, 0])