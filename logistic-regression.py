#%%
from activations import SIGMOID, LINEAR
from network import Network

import numpy as np
from sklearn.datasets import make_classification, make_circles
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

sigmoid = SIGMOID()

#%%
X, y = make_classification(n_features=2, n_informative=1,
    n_redundant=0, n_clusters_per_class=1,
    random_state=42)
Y = np.expand_dims(y, axis=1)
X: np.ndarray
y: np.ndarray
plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0])

#%%
xaxis = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
yaxis = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
xx, yy = np.meshgrid(xaxis, yaxis)
def decision_boundary(clf):
    zz = np.apply_along_axis(clf, 2, np.dstack([xx, yy]))
    plt.contourf(xx, yy, zz, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y)
decision_boundary(lambda x: 0)

#%%
nn = Network([2, 1], activation=LINEAR(),
                     outputActivation=SIGMOID())
losses = nn.fit(X, Y)
decision_boundary(lambda x: nn.predict([x]).item())
plt.title(f'w = {nn.weights[0]} \n b = {nn.biases[1]}')

#%%
plt.plot(range(len(losses)), losses)

#%%
lr = LogisticRegression(penalty='none', random_state=42)
lr.fit(X, y)
decision_boundary(lambda x: lr.predict_proba([x])[0][1])
plt.title(f'w = {lr.coef_} \n b = {lr.intercept_}')


#%%
X, y = make_circles(random_state=42, noise=0.1, factor=0.5)
Y = np.expand_dims(y, axis=1)

#%%
losses = nn.fit(X, Y)
decision_boundary(lambda x: nn.predict([x]).item())
plt.title(f'w = {nn.weights[0]} \n b = {nn.biases[1]}')

#%%
lr.fit(X, y)
decision_boundary(lambda x: lr.predict_proba([x])[0][1])
plt.title(f'w = {lr.coef_} \n b = {lr.intercept_}')

#%%
nn = Network([2, 4, 3, 1])
losses = nn.fit(X, Y)
decision_boundary(lambda x: nn.predict([x]).item())

# #%%
# X_ = np.insert(X, 0, 1, axis=1)
# X_

# #%% [markdown]
# # Instatiate parameter θ as a column vector.

# #%%
# n, p = X.shape
# θ = np.zeros((p + 1, 1))
# θ

# #%% Compute cost
# def compute_cost(X, y, θ):
#     m = len(y)
#     h = sigmoid.func(X @ θ)
#     epsilon = 1e-5
#     return (1/m)*(  ( -y).T @ np.log(  h + epsilon) \
#         -
#                     (1-y).T @ np.log(1-h + epsilon))
# compute_cost(X_, y, θ)

# #%%
# def gradient_descent(X, y, θ, lr=0.03, iters=3000):
#     m = len(y)
#     cost_history = np.zeros((iters, 1))

#     for i in range(iters):
#         θ = θ - (lr/m) * (X.T @ (sigmoid.func(X @ θ) - y))
#         cost_history[i] = compute_cost(X, y, θ).sum().item()
#     return (cost_history, θ)

# (hist, θ_opt) = gradient_descent(X_, y, θ)
# hist.shape, θ_opt.shape

# #%%
# sns.lineplot(x=range(len(hist)), y=hist[:, 0])

# #%%
# θ_opt