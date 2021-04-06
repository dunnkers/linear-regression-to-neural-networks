#%%


#%%
from activations import SIGMOID, LINEAR
from network import Network
import numpy as np
from sklearn.datasets import make_classification, make_circles
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

#%%
X, y = make_classification(n_features=2, n_informative=1,
    n_redundant=0, n_clusters_per_class=1,
    random_state=42)
Y = np.expand_dims(y, axis=1)
X: np.ndarray
y: np.ndarray
plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0])

#%%
def decision_boundary(X, clf):
    xaxis = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    yaxis = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    xx, yy = np.meshgrid(xaxis, yaxis)
    zz = np.apply_along_axis(clf, 2, np.dstack([xx, yy]))
    plt.contourf(xx, yy, zz, alpha=0.4)
decision_boundary(X, lambda x: 0)
plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0])

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
plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0])

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

#%%
from sklearn.datasets import load_diabetes, load_iris
X, y = load_diabetes(return_X_y=True)
X.shape
df = sns.load_dataset('car_crashes')
sns.pairplot(df)

#%%
from sklearn.datasets import load_breast_cancer
data = load_diabetes(as_frame=True)
df = data['frame']
sns.pairplot(df, hue='target')

#%%
penguins = sns.load_dataset('penguins')
sns.pairplot(penguins, hue='species')

#%%
penguins = sns.load_dataset('penguins')
# penguins['is_chinstrap'] = penguins['species'] == 'Chinstrap'
is_chinstrap = lambda species: \
    'Chinstrap' if species == 'Chinstrap' else 'Other'
penguins['Penguin'] = penguins['species'].apply(is_chinstrap)
sns.pairplot(penguins, hue='Penguin')

#%%
sns.scatterplot(data=penguins,
    x='bill_length_mm',
    y='bill_depth_mm',
    hue='Penguin')

#%%
import pandas as pd
X = penguins[['bill_length_mm', 'bill_depth_mm']].values
Y, names = pd.factorize(penguins['Penguin'].values)
Y = np.expand_dims(Y, axis=1)
Y

#%%
nn = Network([2, 3, 1])
losses = nn.fit(X, Y)
decision_boundary(lambda x: nn.predict([x]).item())

#%%
losses

#%%
penguins['species'] == 'Chinstrap'

#%%
from sklearn.datasets import load_iris
data = load_iris(as_frame=True)
df = data['frame']
sns.pairplot(df, hue='target')

#%%
from sklearn.datasets import load_wine
data = load_wine(as_frame=True)
df = data['frame']
sns.pairplot(df, hue='target')

#%%
df['target']


#%%
sns.scatterplot(data=penguins,
    x='bill_length_mm', y='flipper_length_mm',
    hue='species')

#%%
X = penguins[['bill_length_mm', 'flipper_length_mm']].values
Y = penguins[['species']].values
Y

#%%
import pandas as pd
Y, _ = pd.factorize(penguins['species'])
Y
Y = np.expand_dims(Y, axis=1)

#%%
Y = pd.get_dummies(Y).values

#%%
nn = Network([2, 4, 3, 3])
losses = nn.fit(X, Y)

#%%
plt.plot(range(len(losses)), losses)

#%%
decision_boundary(lambda x: nn.predict([x]).item())

# penguins['species']


#%%
X, y = make_