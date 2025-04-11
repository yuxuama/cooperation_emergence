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
    for i in range(n):
        for j in range(n):
            print(adjacency_matrix[i, j], end=", ")
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

def create_random_init(adjacency_matrix, cognitive_capa, min_link):
    """Create a network with `n` random trust link"""
    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            s = np.sum(adjacency_matrix[i])
            remain = cognitive_capa - s
            if remain <= 0:
                adjacency_matrix[i, j] = 0
            else:
                adjacency_matrix[i, j] = np.random.randint(0, min(5 * min_link, remain+1))
              
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


def histogram(trust_adjacency_matrix, parameters, bins=None):
    """Return histogram of the weight distribution for each phenotype and the average distribution in log scale"""
    size = parameters["Community size"]
    if bins is None:
        maxi = int(np.ceil(np.max(trust_adjacency_matrix)))+1
    else:
        maxi = bins.size-1
    phenotype_table = get_vertex_distribution(parameters)

    # Generating each histogram
    phenotype_mean = {"Global": np.zeros(maxi, dtype=float)}
    phenotype_count = {"Global": size}

    for i in range(size):
        data = trust_adjacency_matrix[i]
        n, _ = np.histogram(data, bins=np.arange(int(np.ceil(np.max(data)))+2), density=False)
        n[0] -= 1 # Il faut enlever le fait que la personne n'a pas de lien avec elle-même
        ph = phenotype_table[i]
        if not ph in phenotype_mean:
            phenotype_mean[ph] = np.zeros(maxi, dtype=float)
        if not ph in phenotype_count:
            phenotype_count[ph] = 0
        phenotype_count[ph] += 1
        for i in range(len(n)):
            phenotype_mean[ph][i] += n[i]
            phenotype_mean["Global"][i] += n[i]
    
    for key in phenotype_mean.keys():
        phenotype_mean[key] /= phenotype_count[key] 

    return phenotype_mean

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

def measure_link_asymmetry(link_adjacency_matrix):
    """Return the proportion of link that are asymmetrical"""
    size = link_adjacency_matrix.shape[0]
    number_of_link = 0
    number_of_asymmetric_link = 0

    for i in range(size):
        for j in range(i+1, size):
            if link_adjacency_matrix[i, j] != link_adjacency_matrix[j, i]:
                number_of_asymmetric_link += 1
                number_of_link += 1
            elif link_adjacency_matrix[i, j] > 0:
                number_of_link += 1
    return number_of_asymmetric_link / number_of_link

def measure_individual_asummetry_rate(link_adjacency_matrix):
    """Return the proportion of link that are asymmetrical per vertex"""
    n = link_adjacency_matrix.shape[0]
    number_of_link = np.sum(link_adjacency_matrix, axis=1)
    number_of_asymmetric = np.zeros(n)

    for i in range(n):
        for j in range(i+1, n):
            if link_adjacency_matrix[i, j] != link_adjacency_matrix[j, i]:
                if link_adjacency_matrix[i, j] > 0:
                    number_of_asymmetric[i] += 1
                else:
                    number_of_asymmetric[j] += 1
    return number_of_asymmetric / number_of_link


def measure_saturation_rate(trust_adjacency_matrix, max_load):
    """Return the proportion of edje that are saturated"""
    count = 0
    total = trust_adjacency_matrix.shape[0]

    for i in range(total):
        sumload = np.sum(trust_adjacency_matrix[i])
        if sumload == max_load:
            count += 1

    return count / total

def poisson(k, lamb):
    return np.power(lamb, k) * np.exp(-lamb) / factorial(k)

def factorial(k):
    if k == 0:
        return 1
    return k * factorial(k-1)
