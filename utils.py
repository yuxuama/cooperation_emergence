""" utils.py
All utils functions 
"""

from yaml import safe_load
import numpy as np
import matplotlib.pyplot as plt

def parse_parameters(yaml_file):
    """Load parameters for the simulation from a yaml file"""
    stream = open(yaml_file, 'r')
    return safe_load(stream)

def readable_adjacency(adjacency_matrix=np.ndarray):
    """Gives the adjacency matric in a form usalble in https://graphonline.top/en/create_graph_by_matrix"""
    n = adjacency_matrix.shape[0]
    adj = adjacency_matrix.astype(int)
    for i in range(n):
        for j in range(n):
            print(adj[i, j], end=", ")
        print()

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


def histogram(network):
    """Return the mean histograms of trust: number of link per trust value"""

    mean = np.zeros(network.cognitive_capa+1, dtype=float)
    bins = np.arange(1, network.cognitive_capa+1)

    for v in network.vertices:
        data = np.array(list(v.trust.values()))
        n, _ = np.histogram(data, bins=max(data))
        for i in range(len(n)):
            mean[i] += n[i]
    mean /= network.size

    ax = plt.subplot()
    ax.hist(bins, bins=bins, weights=mean)
    plt.show()
        
if __name__ == "__main__":
    print(parse_parameters(r"parameters.yaml"))
