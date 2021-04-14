# %% [markdown]
# # Neural Network
# Notebook producing the figures in "[Linear Regression to Neural Networks](https://dunnkers.com/linear-regression-to-neural-networks)". Chapter on **Neural Networks**. We are going to fit a Neural Network model to try and classify some Penguin species data. We are going to produce this plot:
#     
# ![neural network fit](./images/neural-fit.gif)
# 
# Let's go!
#%%
import shutil
import tempfile
from time import time
import warnings

import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from IPython import get_ipython
from IPython.display import HTML, set_matplotlib_formats
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from tqdm.notebook import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')
set_matplotlib_formats('svg')
get_ipython().system('python --version')

# %%
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

#%% [markdown]
# Load Penguin dataset.

# %%
data = sns.load_dataset('penguins')
data = data.dropna()
data.dtypes

#%% [markdown]
# Make column for only indicating Chinstrap yes/no.
# %%
peng = lambda x: 'Chinstrap' if x == 'Chinstrap' else 'Other'
data['Penguin'] = data['species'].apply(peng)

# %%
blue_colors = sns.color_palette("Paired", n_colors=2)
sns.scatterplot(data=data_train, x='bill_depth_mm', y='bill_length_mm',
                hue='Penguin', palette=blue_colors)

#%% [markdown]
# Let's try to classify Chinstraps using a Neural Network. We'll use sklearn for this. Try a fit:

# %%
clf = MLPClassifier()
X = data[['bill_depth_mm', 'bill_length_mm']].values
y = data['Penguin'].values
clf.fit(X, y)
clf.n_iter_, clf.classes_, clf.loss_

#%% [markdown]
# Let's get more systematic. First, we'll do a cross-validation training/testing split.

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y,
    stratify=y, random_state=1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

#%% [markdown]
# Next, we create a function to predict our classifier over a grid points, to be used for plotting our decision regions later.
# %%
def apply_over_grid(X, clf, *args):
    xaxis = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    yaxis = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    xx, yy = np.meshgrid(xaxis, yaxis)
    zz = np.dstack([xx, yy])
    ax1, ax2, ax3 = zz.shape
    X = zz.reshape((ax1 * ax2, ax3))
    zz = clf(X, *args)
    zz = zz.reshape((ax1, ax2))
    return xx, yy, zz
xx, yy, zz = apply_over_grid(X, lambda X: X.sum(axis=1))
X.shape, xx.shape, yy.shape, zz.shape

#%% [markdown]
# Wrap our `clf` object in a predictor variable.
#%%
predictor = lambda clf: lambda X: clf.predict_proba(X)[:, 0]
predictor(clf)([[2, 3]])

#%% [markdown]
# Run some couple thousand iterations using our Neural Network. It has a rather 'arbitrary' architecture of 3 layers of 5 nodes each - should be enough to capture the complexity of this dataset.
# %%
max_iter = 25
iterations = 2500
n_fits = iterations // max_iter
clf = MLPClassifier(
    hidden_layer_sizes=(5, 5, 5),
    alpha=0.0005,
    learning_rate_init=0.001,
    max_iter=max_iter,
    n_iter_no_change=max_iter,
    random_state=33,
    tol=1e-15,
    warm_start=True,
    solver='sgd',
    learning_rate='adaptive'
)

records = []
zzz = []
pbar = tqdm(range(n_fits))
iters = 0
for i in pbar:
    clf.fit(X_train, y_train)
    record = { 'Loss': clf.loss_, 'Iteration': clf.n_iter_, 'i': i }
    acc = clf.score(X_train, y_train)
    records.append({ **record, 'Acc': acc, 'Subset': 'Train' })
    acc = clf.score(X_test, y_test)
    records.append({ **record, 'Acc': acc, 'Subset': 'Test' })
    
    _, _, zz = apply_over_grid(X, predictor(clf))
    zzz.append(zz)
    pbar.set_description(f'n_iter={clf.n_iter_}')

#%% [markdown]
# Collect results in a DataFrame and plot loss and accuracy.
#%%
results = pd.DataFrame.from_records(records, index='i')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
sns.lineplot(data=results, x='Iteration', y='Acc', hue='Subset', ax=ax1)
sns.lineplot(data=results, x='Iteration', y='Loss', ax=ax2)


#%% [markdown]
# Plot decision region of last fit.
# %%
def decision_region(ax, zz):
    plt.contourf(xx, yy, zz, alpha=0.8,
                           levels=[0.495, 0.505], colors=['red'])
    plt.contourf(xx, yy, zz, alpha=0.4, cmap='Blues')
    sns.scatterplot(data=data, x='bill_depth_mm', y='bill_length_mm',
                    hue='Penguin', palette=blue_colors, ax=ax)
    
fig = plt.figure()
ax = fig.gca()
decision_region(ax, zzz[-1])
plt.colorbar()

#%% [markdown]
# We are going to create a range of images. Create a temporary folder for them.
# %%
folder = tempfile.mkdtemp()
print(f'Saving images to {folder}')

# %%
frames = 100
assert(frames <= n_fits)
interval = n_fits // frames
print(f'{n_fits} fits, {frames} frames; so interval of {interval}.')


#%% [markdown]
# Compute and save separate GIF images.
#%%
pbar = tqdm(total=frames)
def create_frame(frame, ax, zzz):
    ax.cla()

    i = frame * interval

    # decision region
    decision_region(ax, zzz[i])

    # plot title
    result = results[results.index == i]
    trn = result[result['Subset'] == 'Train']
    tst = result[result['Subset'] == 'Test']
    loss = result['Loss'].unique().item()
    acc, acc_test = trn['Acc'].item(), tst['Acc'].item()
    iteration = result['Iteration'].unique().item()
    plt.title(f'Neural Network fit iteration {iteration}\n'+
              f'Acc: train={acc:.3f}, test={acc_test:.3f}, '+
              f'log loss: train={loss:.3f}')
    plt.savefig(f'{folder}/frame_{frame:03}.png')

    # progress bar
    pbar.update()
    
fig = plt.figure()
ax = fig.gca()
animation = FuncAnimation(fig, create_frame,
    frames=frames, fargs=(ax, zzz), interval=100) # => 10 fps
animated = animation.to_jshtml()
pbar.close()


#%% [markdown]
# Show video using Jupyter HTML widget.
# %%
HTML(animated)

#%% [markdown]
# Convert separate images into a GIF.
# %%
name = './images/neural-fit.gif'
get_ipython().system('convert -background white -alpha remove '+
    '-dispose Previous +antialias -layers OptimizePlus '+
    f'{folder}/*.png {name}')


#%% [markdown]
# Clean the temporary folder 💎
# %%
shutil.rmtree(folder)


