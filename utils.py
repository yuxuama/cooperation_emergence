""" utils.py
All utils functions 
"""

from yaml import safe_load, dump
import numpy as np
import matplotlib.pyplot as plt

def parse_parameters(yaml_file):
    """Load parameters for the simulation from a yaml file"""
    stream = open(yaml_file, 'r')
    return safe_load(stream)

def save_parameters(data, dir_path):
    """Save the parameters represented by `data` in the `dir_path` directory"""
    stream = open(dir_path + "parameters.yaml", 'w')
    dump(data, stream=stream,default_flow_style=False)

def readable_adjacency(adjacency_matrix=np.ndarray):
    """Gives the adjacency matric in a form usalble in https://graphonline.top/en/create_graph_by_matrix"""
    n = adjacency_matrix.shape[0]
    adj = adjacency_matrix.astype(int)
    for i in range(n):
        for j in range(n):
            print(adj[i, j], end=", ")
        print()

def get_vertex_distribution(parameters):
    """Give the link vertex index - vertex phenotype according to the parameters"""
    strategy_distrib = parameters["Strategy distributions"]
    size = parameters["Community size"]
    phenotypes = list(strategy_distrib.keys())
    
    distribution_grid = [0] # Used to initialized populations of each phenotypes according to parameters.
    for p in strategy_distrib.values():
        distribution_grid.append(distribution_grid[-1] + p)
    distribution_grid = size * np.array(distribution_grid)

    pointer = 0
    table = ["" for _ in range(size)]
    for i in range(size):
        while i > distribution_grid[pointer]:
            pointer += 1
        table[i] = phenotypes[pointer-1]
    
    return table

def create_social_group(size, assignation, adjacency_matrix, min_trust):
    """Create a fully connected graph of size `size` among an unassigned group of people
    Allow to create disconnected social group"""
    remaining = assignation.size - np.sum(assignation)
    
    sample = np.random.choice(remaining, size, replace=False)
    sample = np.sort(sample)
    real_sample = np.zeros(size, dtype=int)

    count = 0
    s_pointer = 0
    for i in range(assignation.size):
        if not assignation[i]:
            if sample[s_pointer] == count:
                real_sample[s_pointer] = i
                s_pointer += 1
                if s_pointer >= size:
                    break
            count += 1
    for i in range(size):
        assignation[real_sample[i]] = True
        for j in range(size):
            if j != i:
                adjacency_matrix[real_sample[i], real_sample[j]] = min_trust

def plot(layout, ax, i, j, data, bins, title, log):
    """Auxiliary function used in histogram"""
    if layout[0] > 1:
        if log:
            ax[i, j].plot(np.log(data), '+')
        else:
            ax[i, j].hist(bins, bins=bins, weights=data)
        ax[i, j].set_title(title)
    else:
        if log:
            ax[j].plot(np.log(data), '+')
        else:
            ax[j].hist(bins, bins=bins, weights=data)
        ax[j].set_title(title)


def histogram(trust_adjacency_matrix, parameters):
    """Return histogram of the weight distribution for each phenotype and the average distribution in log scale"""
    size = parameters["Community size"]
    maxi = int(np.max(trust_adjacency_matrix))
    phenotype_table = get_vertex_distribution(parameters)

    # Generating each histogram
    phenotype_mean = {}
    phenotype_count = {}

    mean = np.zeros(maxi+1, dtype=float)
    bins = np.arange(maxi+1)

    for i in range(size):
        data = trust_adjacency_matrix[i]
        n, _ = np.histogram(data, bins=int(np.max(data)+1), density=False)
        n[0] -= 1 # Il faut enlever le fait que la personne n'a pas de lien avec elle-même
        ph = phenotype_table[i]
        if not ph in phenotype_mean:
            phenotype_mean[ph] = np.zeros(maxi+1, dtype=float)
        if not ph in phenotype_count:
            phenotype_count[ph] = 0
        phenotype_count[ph] += 1
        for i in range(len(n)):
            phenotype_mean[ph][i] += n[i]
            mean[i] += n[i]
    
    for key in phenotype_mean.keys():
        phenotype_mean[key] /= phenotype_count[key] 
    
    mean /= size

    return phenotype_mean, mean

def plot_histogram(phenotype_mean, mean, parameters, log=False):
    possible_phenotype = list(parameters["Strategy distributions"].keys())
    bins = np.arange(mean.size)
    fig_layout = [(1, 2), (1, 3), (2, 3), (2, 3), (2, 3)]
    layout = fig_layout[len(possible_phenotype) - 1]
    fig, ax = plt.subplots(layout[0], layout[1], sharex=True)
    for j in range(layout[1]):
        for i in range(layout[0]):
            if layout[1]*i + j < len(possible_phenotype):
                ph = possible_phenotype[3*i+j]
                plot(layout, ax, i, j, phenotype_mean[ph], bins, ph, log)
            elif layout[1]*i + j == len(possible_phenotype):
                plot(layout, ax, i, j, mean, bins, "Average", True)
            else:
                if layout[0] > 1:
                    ax[i, j].remove()
                else:
                    ax[j].remove()
    
    return fig, ax

def measure_link_asymmetry(link_adjacency_matric):
    """Return the proportion of link that are asymmetrical"""
    size = link_adjacency_matric.shape[0]
    number_of_link = size * (size - 1) / 2
    number_of_asymmetric_link = 0

    for i in range(size):
        for j in range(i+1, size):
            if link_adjacency_matric[i, j] != link_adjacency_matric[j, i]:
                number_of_asymmetric_link += 1
    
    return number_of_asymmetric_link / number_of_link

def poisson(k, lamb):
    return np.power(lamb, k) * np.exp(-lamb) / factorial(k)

def factorial(k):
    if k == 0:
        return 1
    return k * factorial(k-1)
