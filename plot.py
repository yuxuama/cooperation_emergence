"""plot.py

File where plot routines are defined
"""

import numpy as np
import matplotlib.pyplot as plt
from analysis import measure, estimate_etas

def plot_sub(layout, ax, i, j, data, bins, title, log, **kwargs):
    """Auxiliary function used in histogram"""
    n = np.arange(data.size)
    if layout[0] > 1:
        if log:
            ax[i, j].plot(np.log(data), '+', **kwargs)
        else:
            ax[i, j].hist(n, bins=bins, weights=data, **kwargs)
        ax[i, j].set_title(title)
    else:
        if log:
            ax[j].plot(np.log(data), '+', **kwargs)
        else:
            ax[j].hist(n, bins=bins, weights=data, **kwargs)
        ax[j].set_title(title)

def plot_one_histogram(ax, data, bins=None, **kwargs):
    n = np.arange(data.size)
    if bins is None:
        bins = np.arange(data.size + 1)
    plot = ax.hist(n, bins=bins, weights=data, **kwargs)
    return ax, plot

def plot_histogram(phenotype_mean, parameters, log=False, **kwargs):
    possible_phenotype = list(parameters["Strategy distributions"].keys())
    bins = np.arange(phenotype_mean["Global"].size+1)
    fig_layout = [(1, 2), (1, 3), (2, 3), (2, 3), (2, 3)]
    layout = fig_layout[len(possible_phenotype) - 1]
    fig, ax = plt.subplots(layout[0], layout[1], sharex=True)
    for j in range(layout[1]):
        for i in range(layout[0]):
            if layout[1]*i + j < len(possible_phenotype):
                ph = possible_phenotype[3*i+j]
                plot_sub(layout, ax, i, j, phenotype_mean[ph], bins, ph, log, **kwargs)
            elif layout[1]*i + j == len(possible_phenotype):
                plot_sub(layout, ax, i, j, phenotype_mean["Global"], bins, "log Average", True, **kwargs)
            else:
                if layout[0] > 1:
                    ax[i, j].remove()
                else:
                    ax[j].remove()
    
    return fig, ax

def plot_evolution(ax, quantity, oper, start, end, step, parameters, **kwargs):
    x = np.arange(start, end, step)
    data = np.zeros(x.size)
    t, l = oper.resolve(x[0])
    for i in range(data.size):
        data[i] = measure(quantity, t, l, parameters)
        t, l = oper.resolve(x[i])
    
    ax.plot(x, data, **kwargs)
    ax.set_xlabel("Interaction")
    ax.set_ylabel(quantity)
    
def plot_randomized_evolution(ax, quantity, oper, start, end, step, parameters, niter=250, mode="o", mc_iter=10, **plot_kwargs):
    x = np.arange(start, end, step)
    data = np.zeros(x.size)
    t, l = oper.resolve(x[0])
    for i in range(data.size):
        data[i] = measure(quantity, t, l, parameters, random=True, niter=niter, mode=mode, mc_iter=mc_iter)
        t, l = oper.resolve(x[i])
    
    ax.plot(x, data, **plot_kwargs)
    ax.set_xlabel("Interaction")
    ax.set_ylabel(quantity)