# %%
from IPython import get_ipython

# %% [markdown]
# # Logistic Regression
# Notebook producing the figures in "[Linear Regression to Neural Networks](https://dunnkers.com/linear-regression-to-neural-networks)". Chapter on **Logistic Regression**. We are going to fit a Logistic Regression model to try and classify some Penguin species data. We are going to produce this plot:
#     
# ![logistic fit](./images/logistic-fit.gif)
# 
# Let's go!

# %%
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from IPython.display import HTML
import IPython
from matplotlib.animation import FuncAnimation
from time import time
from tqdm.notebook import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import tempfile
import shutil
get_ipython().run_line_magic('matplotlib', 'inline')
set_matplotlib_formats('svg')
get_ipython().system('python --version')


# %%
from sklearn.exceptions import ConvergenceWarning
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


# %%
data = sns.load_dataset('penguins')
data = data.dropna()
data.dtypes

# %%
peng = lambda x: 'Chinstrap' if x == 'Chinstrap' else 'Other'
data['Penguin'] = data['species'].apply(peng)

# %%
blue_colors = sns.color_palette("Paired", n_colors=2)
sns.scatterplot(data=data_train, x='bill_depth_mm', y='bill_length_mm',
                hue='Penguin', palette=blue_colors)

# %%
clf_args = {
    'hidden_layer_sizes': (5, 5, 5),
    'alpha': 0.0,
    'learning_rate_init': 0.001,
    'max_iter': 50,
    'random_state': 33,
    'tol': 1e-15,
    'warm_start': True
}
clf = MLPClassifier(**clf_args)

X = data[['bill_depth_mm', 'bill_length_mm']].values
y = data['Penguin'].values

clf.fit(X, y)
classes = clf.classes_
clf.coefs_, clf.intercepts_, clf.n_iter_, clf.classes_, clf.loss_

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y,
    stratify=y, random_state=1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# %%
max_iter = 50
iterations = 3000
n_fits = iterations // max_iter
clf = MLPClassifier(**clf_args)

records = []
pbar = tqdm(range(n_fits))
iters = 0
for i in pbar:
    clf.fit(X_train, y_train)
    iters += clf.n_iter_
    record = { 'Loss': clf.loss_, 'Iteration': iters }
    acc = clf.score(X_train, y_train)
    records.append({ **record, 'Acc': acc, 'Subset': 'Train' })
    acc = clf.score(X_test, y_test)
    records.append({ **record, 'Acc': acc, 'Subset': 'Test' })
    
#%%
clf.get_params()

#%%
results = pd.DataFrame.from_records(records)
results

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
sns.lineplot(data=results, x='Iteration', y='Acc', hue='Subset', ax=ax1)
sns.lineplot(data=results, x='Iteration', y='Loss', ax=ax2)


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

#%%
# def predictor(clf):
clf.predict_proba(X)[:, 1]


# %%
def decision_region(ax):
    # predictor = lambda X, coef: predict_proba_lr(add_bias(X), coef)
    predictor = lambda X: clf.predict_proba(X)[:, 1]
    vals = data[['bill_depth_mm', 'bill_length_mm']].values
    xx, yy, zz = apply_over_grid(vals, predictor)
    plt.contourf(xx, yy, zz, alpha=0.7,
                           levels=[0.495, 0.505], colors=['red'])
    plt.contourf(xx, yy, zz, alpha=0.4, cmap='Blues')
    sns.scatterplot(data=data, x='bill_depth_mm', y='bill_length_mm',
                    hue='Penguin', palette=blue_colors, ax=ax)
    
fig = plt.figure()
ax = fig.gca()
decision_region(ax)
plt.colorbar()
# f"loss={losses[-1]}, acc={accs[-1]}"


# %%
folder = tempfile.mkdtemp()
print(f'Saving images to {folder}')


# %%
import time

frames = 200
iter_right_lim = 20000
n_fits_show = iter_right_lim // max_iter
intervals = n_fits_show // frames
pbar = tqdm(total=frames)
def create_frame(frame, ax, coefs):
    ax.cla()
    i = frame * intervals
    coef = coefs[i]
    loss = losses[i][0]
    loss_test = losses[i][1]
    acc = accs[i][0]
    acc_test = accs[i][1]
    iteration = iters[i]
    
    decision_region(coef, ax)

    plt.title(f'Logistic Regression fit iteration {iteration}\n'+
              f'Acc: train={acc:.2f}, test={acc_test:.2f}, '+
              f'log loss: train={loss:.2f}, test={loss_test:.2f}')
    pbar.update()
    plt.savefig(f'{folder}/frame_{frame:03}.png')
    
fig = plt.figure()
ax = fig.gca()
animation = FuncAnimation(fig, create_frame, frames=frames,
                          fargs=(ax, coefs), interval=100)
                # Interval at 1000/100 = 10 frames per second
animated = animation.to_jshtml()
pbar.close()


# %%
HTML(animated)


# %%
get_ipython().system('convert -background white -alpha remove -dispose Previous +antialias -layers OptimizePlus $folder/*.png ./images/logistic-fit.gif')


# %%
shutil.rmtree(folder)


