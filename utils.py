""" utils.py
All utils functions 
"""

from yaml import safe_load, dump
import numpy as np
import matplotlib.pyplot as plt

"""UTILS"""

def parse_parameters(yaml_file):
    """Load parameters for the simulation from a yaml file"""
    stream = open(yaml_file, 'r')
    return safe_load(stream)

def save_parameters(data, dir_path):
    """Save the parameters represented by `data` in the `dir_path` directory"""
    stream = open(dir_path + "parameters.yaml", 'w')
    dump(data, stream=stream,default_flow_style=False)

def print_parameters(parameters):
    """Print the parameters extracted from a yaml file"""
    for key, values in parameters.items():
        print("..", key, ": ", values)

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

"""INIT functions"""

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
                adjacency_matrix[i, j] = np.random.randint(0, min(2 * min_link, remain+1))

