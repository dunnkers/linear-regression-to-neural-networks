#%% [markdown]
# # From Linear Regression to Neural Networks
# These days there exists much hype around sophisticated machine learning methods such as Neural Networks. However, we do not always require the full complexity of a Neural Network: sometimes, a simpler model will do the job just fine. In this project, we take a journey starting from the most fundamental statistical machinery to model data distributions - linear regression and logistic classification - and then explain the benefits of constructing more complex models, such as Support Vector Machines and Neural Networks. We will not shy away from the math, whilst still working with tangible examples at all times: we will work with real world datasets and we'll get to apply our models in code right in this document as we go on. Let's start!



#%%
import numpy as np
import seaborn as sns
import sklearn
import pandas as pd
from network import Network
from activations import LINEAR, RELU, SIGMOID, TANH

#%%
penguins = sns.load_dataset('penguins')
is_chinstrap = lambda species: \
    'Chinstrap' if species == 'Chinstrap' else 'Other'
penguins['Penguin'] = penguins['species'].apply(is_chinstrap)
sns.pairplot(penguins, hue='Penguin')

#%%
penguins = penguins.dropna(axis=0, how='any')
X = penguins[['bill_length_mm', 'bill_depth_mm']].values
Y, names = pd.factorize(penguins['Penguin'].values)
Y = np.expand_dims(Y, axis=1)
X.shape, Y.shape


#%%
sns.scatterplot(data=penguins,
    x='bill_length_mm',
    y='bill_depth_mm',
    hue='Penguin')

#%%
nn = Network([2, 5, 2, 1], activation=TANH(),
    outputActivation=TANH())
losses = nn.fit(X, Y, lr=0.005, max_epochs=10000)

#%%
sns.lineplot(x=range(len(losses)), y=losses)

#%%
def decision_boundary(X, clf):
    xaxis = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    yaxis = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    xx, yy = np.meshgrid(xaxis, yaxis)
    zz = np.apply_along_axis(clf, 2, np.dstack([xx, yy]))
    plt.contourf(xx, yy, zz, alpha=0.4)
decision_boundary(X, lambda x: nn.predict([x]).item())
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y[:, 0])

#%%
nn.predict([X[0]]).item()

#%%
nn.forward(X[1])
nn.backward(Y[1])
nn.learn()
nn.get_loss(Y[1])
#%%
len(losses)
# nn.biases

#%% [markdown]
# ## Citations
# - [Gorman KB, Williams TD, Fraser WR (2014). Ecological sexual dimorphism and environmental variability within a community of Antarctic penguins (genus Pygoscelis). PLoS ONE 9(3):e90081.](https://doi.org/10.1371/journal.pone.0090081)