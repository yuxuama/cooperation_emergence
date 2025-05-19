"""plot.py

File where plot routines are defined
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from analysis import measure_global
from scipy.optimize import curve_fit

################################################################################################
# Plot trust distribution histograms
################################################################################################


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
    fig, ax = plt.subplots(layout[0], layout[1], sharex=True, figsize=(8, 6))
    for j in range(layout[1]):
        for i in range(layout[0]):
            if layout[1]*i + j < len(possible_phenotype):
                ph = possible_phenotype[3*i+j]
                plot_sub(layout, ax, i, j, phenotype_mean[ph], bins, ph, log, **kwargs)
            elif layout[1]*i + j == len(possible_phenotype):
                plot_sub(layout, ax, i, j, phenotype_mean["Global"], bins, "Average", log, **kwargs)
            else:
                if layout[0] > 1:
                    ax[i, j].remove()
                else:
                    ax[j].remove()
    
    return fig, ax

################################################################################################
# Plot xhis
################################################################################################

def plot_xhi_by_phenotype(xhi_means, **plot_kwargs):
    """Plot all mean xhis for each phenotype and the corresponding eta fit"""
    possible_phenotype = list(xhi_means.keys())
    possible_phenotype.append(possible_phenotype.pop(0))
    fig_layout = [(1, 2), (1, 3), (2, 3), (2, 3), (2, 3)]
    layout = fig_layout[len(possible_phenotype) - 2]
    fig, ax = plt.subplots(layout[0], layout[1], figsize=(8, 6))
    model = lambda j, eta: (np.exp(eta * j) - 1) / (np.exp(eta) - 1)
    for i in range(layout[0]):
        for j in range(layout[1]):
            if layout[0] == 1:
                selector = j
            else:
                selector = (i, j)
            index = i * layout[1] + j
            ph = possible_phenotype[index]
            xhi = xhi_means[ph]
            t_norm = np.arange(xhi.size) / (xhi.size - 1)
            popt, _ = curve_fit(model, t_norm, xhi)

            ax[selector].plot(t_norm, xhi, "+", color="tab:blue", label="Mean", **plot_kwargs)
            ax[selector].plot(t_norm, model(t_norm, popt[0]), color="tab:orange", label="Fit ($\eta$ = {})".format(round(popt[0], 2)))
            ax[selector].set_title(ph)
            ax[selector].legend()
    fig.suptitle("Average xhi by phenotype")
    return ax

################################################################################################
# Plots with datasets
################################################################################################

def plot_hist_by_phenotype(dataset, quantity, **plot_kwargs):
    """Generate appropriate layour for phenotype ploting"""
    dta = dataset.group_by("Phenotype").aggregate(quantity)
    global_dta = dataset.aggregate(quantity)
    possible_phenotype = sorted(list(dta.keys()))
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
            data = list(data.values())
            n, _, _ = ax[selector].hist(data, **plot_kwargs)
            maxi = np.max(n)
            median = np.median(data)
            ax[selector].vlines(median, 0, maxi*1.1, colors="k", linestyles="dashed", label="Median: {}".format(round(median, 2)))
            ax[selector].set_title(title)
            ax[selector].set_ylim([0, maxi*1.1])
            ax[selector].legend()
    fig.suptitle("Histogram of {0} @ inter {1}".format(quantity, dataset.niter))
    return ax

def plot_diadic_pattern(dataset, **plot_kwargs):
    """Plot bar graph with diadic pattern repartition per phenotype"""
    dta = dataset.group_by("Phenotype").aggregate((".", "<-", "->", "--"))
    global_dta = dataset.aggregate((".", "<-", "->", "--"))
    possible_phenotype = sorted(list(dta.keys()))
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
            ax[selector].set_ylabel("Occurences")
    fig.suptitle("Diadic pattern @ iter {}".format(dataset.niter))
    return ax

def plot_bar_diadic_pattern(dataset, **plot_kwargs):
    dta = dataset.group_by("Phenotype").aggregate((".", "<-", "->", "--"))
    possible_phenotype = sorted(list(dta.keys()))
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
    ax.set_ylabel("Occurences")
    plt.legend()

def plot_triadic_pattern(triadic_dataset, selector="Number", **plot_kwargs):
    data = triadic_dataset.aggregate(selector).get_item(selector).get_all_item()
    triangle_names = list(data.keys())

    def get_image(name):
        path = r"C:\Users\Matthieu\Documents\_Travail\Stages\Stage M1\Workspace\image\triadic_{}.png".format(name)
        im = plt.imread(path)
        return im

    def offset_image(coord, name, ax):
        img = get_image(name)
        im = OffsetImage(img, zoom=0.2)
        im.image.axes = ax
        ab = AnnotationBbox(im, (coord, 0),  xybox=(0., -16.), frameon=False,
                            xycoords='data',  boxcoords="offset points", pad=0)
        ax.add_artist(ab)

    _, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.bar(range(0, len(data)*2, 2), list(data.values()), width=1, align="center", **plot_kwargs)
    ax.tick_params(axis='x', which='both', labelbottom=False, top=False, bottom=False)
    ax.set_ylabel("Occurences")
    if selector == "Number":
        title = "Global"
    else:
        title = selector
    ax.set_title("Triadic pattern frequency for {0} @ inter {1}".format(selector, triadic_dataset.niter))

    for i in range(0, len(data)*2, 2):
        offset_image(i, triangle_names[i//2], ax)
    
    return ax

def plot_triadic_pattern_phenotype(triadic_dataset, parameters, **plot_kwargs):
    possible_fields = list(parameters["Strategy distributions"].keys())
    possible_fields.append("Number")
    data = triadic_dataset.aggregate(possible_fields)
    triangle_names = list(data.get_item("Number").get_all_item().keys())

    def get_image(name):
        path = r"C:\Users\Matthieu\Documents\_Travail\Stages\Stage M1\Workspace\image\triadic_{}.png".format(name)
        im = plt.imread(path)
        return im

    def offset_image(coord, name, ax):
        img = get_image(name)
        im = OffsetImage(img, zoom=0.2)
        im.image.axes = ax
        ab = AnnotationBbox(im, (coord, 0),  xybox=(0., -16.), frameon=False,
                            xycoords='data',  boxcoords="offset points", pad=0)
        ax.add_artist(ab)

    _, ax = plt.subplots(1, 1, figsize=(8, 5))

    bottom = np.zeros(16)
    for i in range(len(possible_fields) - 1):
        ph_data = data.get_item(possible_fields[i]).get_all_item()
        values = np.array(list(ph_data.values())) / 3
        ax.bar(range(0, len(ph_data)*2, 2), values, width=1, align="center", bottom=bottom, label=possible_fields[i], **plot_kwargs)
        bottom += values
    ax.tick_params(axis='x', which='both', labelbottom=False, top=False, bottom=False)
    ax.set_ylabel("Occurences")
    ax.set_title("Triadic pattern frequency @ inter {}".format(triadic_dataset.niter))
    ax.legend()

    for i in range(0, len(ph_data)*2, 2):
        offset_image(i, triangle_names[i//2], ax)
    
    return ax

################################################################################################
# Plot evolution
################################################################################################

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