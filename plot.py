"""plot.py

File where plot routines are defined
"""

import numpy as np
import matplotlib.pyplot as plt
from analysis import measure_global

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

def plot_hist_by_phenotype(dataset, quantity, **plot_kwargs):
    """Generate appropriate layour for phenotype ploting"""
    dta = dataset.group_by("Phenotype").aggregate(quantity)
    global_dta = dataset.aggregate(quantity)
    possible_phenotype = list(dta.keys())
    fig_layout = [(1, 2), (1, 3), (2, 3), (2, 3), (2, 3)]
    layout = fig_layout[dta.size - 1]
    fig, ax = plt.subplots(layout[0], layout[1], figsize=(8, 6))
    for i in range(layout[0]):
        for j in range(layout[1]):
            if layout[0] == 1:
                selector = j
            else:
                selector = (i, j)
            index = i * layout[1] + j
            if index == dta.size:
                data = global_dta.get_item(quantity).get_all_item()
                title = "Global"
            else:
                data = dta.get_item(possible_phenotype[index]).get_item(quantity).get_all_item()
                title = possible_phenotype[index]
            ax[selector].hist(list(data.values()), **plot_kwargs)
            ax[selector].set_title(title)
    fig.suptitle("Histogram of {0} @ inter {1}".format(quantity, dataset.niter))
    return ax

def plot_diadic_pattern(dataset, **plot_kwargs):
    """Plot bar graph with diadic pattern repartition per phenotype"""
    dta = dataset.group_by("Phenotype").aggregate((".", "<-", "->", "--"))
    global_dta = dataset.aggregate((".", "<-", "->", "--"))
    possible_phenotype = list(dta.keys())
    fig_layout = [(1, 2), (1, 3), (2, 3), (2, 3), (2, 3)]
    layout = fig_layout[dta.size - 1]
    fig, ax = plt.subplots(layout[0], layout[1], figsize=(10, 6))
    for i in range(layout[0]):
        for j in range(layout[1]):
            if layout[0] == 1:
                selector = j
            else:
                selector = (i, j)
            index = i * layout[1] + j
            if index == dta.size:
                no_link = sum(global_dta.get_item(".").get_all_item().values())
                in_link = sum(global_dta.get_item("<-").get_all_item().values())
                out_link = sum(global_dta.get_item("->").get_all_item().values())
                bi_link = sum(global_dta.get_item("--").get_all_item().values())
                data = [no_link, in_link, out_link, bi_link]
                title = "Global"
            else:
                no_link = sum(dta.get_item(possible_phenotype[index]).get_item(".").get_all_item().values())
                in_link = sum(dta.get_item(possible_phenotype[index]).get_item("<-").get_all_item().values())
                out_link = sum(dta.get_item(possible_phenotype[index]).get_item("->").get_all_item().values())
                bi_link = sum(dta.get_item(possible_phenotype[index]).get_item("--").get_all_item().values())
                data = [no_link, in_link, out_link, bi_link]
                title = possible_phenotype[index]
            ax[selector].bar(["0", "<--", "-->", "<->"], data, **plot_kwargs)
            ax[selector].set_title(title)
    fig.suptitle("Diadic pattern @ iter {}".format(dataset.niter))
    return ax

def plot_bar_diadic_pattern(dataset, **plot_kwargs):
    dta = dataset.group_by("Phenotype").aggregate((".", "<-", "->", "--"))
    possible_phenotype = list(dta.keys())
    link_type = ["0", "<--", "-->", "<->"]
    bottom = np.zeros(4)
    ax = plt.subplot(1, 1, 1)
    for i in range(len(possible_phenotype)):
        no_link = sum(dta.get_item(possible_phenotype[i]).get_item(".").get_all_item().values())
        in_link = sum(dta.get_item(possible_phenotype[i]).get_item("<-").get_all_item().values())
        out_link = sum(dta.get_item(possible_phenotype[i]).get_item("->").get_all_item().values())
        bi_link = sum(dta.get_item(possible_phenotype[i]).get_item("--").get_all_item().values())
        data = np.array([no_link, in_link, out_link, bi_link])
        ax.bar(link_type, data, bottom=bottom, label=possible_phenotype[i], **plot_kwargs)
        bottom += data
    ax.set_title("Diadic pattern @ inter {}".format(dataset.niter))
    plt.legend()

def plot_evolution(ax, quantity, oper, start, end, step, parameters, **kwargs):
    x = np.arange(start, end, step)
    data = np.zeros(x.size)
    t, l = oper.resolve(x[0])
    for i in range(data.size):
        data[i] = measure_global(quantity, t, l, parameters)
        t, l = oper.resolve(x[i])
    
    ax.plot(x, data, **kwargs)
    ax.set_xlabel("Interaction")
    ax.set_ylabel(quantity)
    
def plot_randomized_evolution(ax, quantity, oper, start, end, step, parameters, niter=300, mode="o", mc_iter=50, **plot_kwargs):
    x = np.arange(start, end, step)
    data = np.zeros(x.size)
    t, l = oper.resolve(x[0])
    for i in range(data.size):
        data[i] = measure_global(quantity, t, l, parameters, random=True, niter=niter, mode=mode, mc_iter=mc_iter)
        t, l = oper.resolve(x[i])
    
    ax.plot(x, data, **plot_kwargs)
    ax.set_xlabel("Interaction")
    ax.set_ylabel(quantity)