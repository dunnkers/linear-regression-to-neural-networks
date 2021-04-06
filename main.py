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
X = penguins[['bill_length_mm', 'bill_depth_mm']].values
Y, names = pd.factorize(penguins['Penguin'].values)
Y = np.expand_dims(Y, axis=1)
X.shape, Y.shape

#%%
nn = Network([2, 3, 1], activation=SIGMOID(),
    outputActivation=LINEAR())
losses = nn.fit(X, Y)
decision_boundary(lambda x: nn.predict([x]).item())

#%%
losses