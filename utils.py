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

def create_social_groupe(size, assignation, adjacency_matrix, min_trust):
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


def histogram(trust_adjacency_matrix, parameters):
    """Return histogram of the weight distribution for each phenotype and the average distribution in log scale"""
    fig_layout = [(1, 2), (1, 3), (2, 3), (2, 3), (2, 3)]

    size = parameters["Community size"]
    maxi = int(np.max(trust_adjacency_matrix))
    phenotype_table = get_vertex_distribution(parameters)
    possible_phenotype = list(parameters["Strategy distributions"].keys())
    layout = fig_layout[len(possible_phenotype) - 1]
    print(layout)

    # Generating each histogram
    phenotype_mean = {}

    mean = np.zeros(maxi+1, dtype=float)
    bins = np.arange(maxi+1)

    for i in range(size):
        data = trust_adjacency_matrix[i]
        n, _ = np.histogram(data, bins=int(np.max(data)))
        ph = phenotype_table[i]
        phenotype_mean[ph] = np.zeros(maxi+1, dtype=float)
        for i in range(len(n)):
            phenotype_mean[ph][i] += n[i]
            mean[i] += n[i]
    mean /= size
    
    fig, ax = plt.subplots(layout[0], layout[1], sharex=True)
    for j in range(layout[1]):
        for i in range(layout[0]):
            if layout[1]*i + j < len(possible_phenotype):
                ph = possible_phenotype[3*i+j]
                if layout[0] > 1:
                    ax[i, j].hist(bins, bins=bins, weights=phenotype_mean[ph])
                    ax[i, j].set_title(ph)
                else:
                    ax[j].hist(bins, bins=bins, weights=phenotype_mean[ph])
                    ax[j].set_title(ph)
            elif layout[1]*i + j == len(possible_phenotype):
                if layout[0] > 1:
                    ax[i, j].plot(np.log(mean), '+')
                    ax[i, j].set_title("Average")
                else:
                    ax[j].plot(np.log(mean), '+')
                    ax[j].set_title("Average")
            else:
                if layout[0] > 1:
                    ax[i, j].remove()
                else:
                    ax[j].remove()
    
    plt.show()