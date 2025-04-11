"""plot.py

File where plot routines are defined
"""

import numpy as np
import matplotlib.pyplot as plt
from analysis import measure_saturation_rate, measure_link_asymmetry

def plot(layout, ax, i, j, data, bins, title, log):
    """Auxiliary function used in histogram"""
    n = np.arange(data.size)
    if layout[0] > 1:
        if log:
            ax[i, j].plot(np.log(data), '+')
        else:
            ax[i, j].hist(n, bins=bins, weights=data)
        ax[i, j].set_title(title)
    else:
        if log:
            ax[j].plot(np.log(data), '+')
        else:
            ax[j].hist(n, bins=bins, weights=data)
        ax[j].set_title(title)

def plot_unique_hist(ax, data, bins=None):
    n = np.arange(data.size)
    if bins is None:
        bins = np.arange(data.size + 1)
    plot = ax.hist(n, bins=bins, weights=data)
    return ax, plot

def plot_histogram(phenotype_mean, parameters, log=False):
    possible_phenotype = list(parameters["Strategy distributions"].keys())
    bins = np.arange(phenotype_mean["Global"].size+1)
    fig_layout = [(1, 2), (1, 3), (2, 3), (2, 3), (2, 3)]
    layout = fig_layout[len(possible_phenotype) - 1]
    fig, ax = plt.subplots(layout[0], layout[1], sharex=True)
    for j in range(layout[1]):
        for i in range(layout[0]):
            if layout[1]*i + j < len(possible_phenotype):
                ph = possible_phenotype[3*i+j]
                plot(layout, ax, i, j, phenotype_mean[ph], bins, ph, log)
            elif layout[1]*i + j == len(possible_phenotype):
                plot(layout, ax, i, j, phenotype_mean["Global"], bins, "log Average", True)
            else:
                if layout[0] > 1:
                    ax[i, j].remove()
                else:
                    ax[j].remove()
    
    return fig, ax

def plot_saturation_evolution(ax, oper, start, end, step, cognitive_cap, **kwargs):
    x = np.arange(start//step, end//step)
    data = np.zeros(end//step - start//step)
    t, _ = oper.resolve(start)
    last = start
    for i in range(0, end//step - start//step):
        data[i] = measure_saturation_rate(t, cognitive_cap)
        last += step
        t, _ = oper.resolve(last)
    ax.plot(step * x, data, **kwargs)
    ax.set_xlabel("Interaction")
    ax.set_ylabel("Global saturation rate")

def plot_asymmetry_evolution(ax, oper, start, end, step, **kwargs):
    x = np.arange(start//step, end//step)
    data = np.zeros(end//step - start//step)
    _, l = oper.resolve(start)
    last = start
    for i in range(0, end//step - start//step):
        data[i] = measure_link_asymmetry(l)
        last += step
        _, l = oper.resolve(last)
    ax.plot(step * x, data, **kwargs)
    ax.set_xlabel("Interaction")
    ax.set_ylabel("Global asymmetry rate")