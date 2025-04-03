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

def readable_adjency(adjency_matrix=np.ndarray):
    """Gives the adjency matric in a form usalble in https://graphonline.top/en/create_graph_by_matrix"""
    n = adjency_matrix.shape[0]
    adj = adjency_matrix.astype(int)
    for i in range(n):
        for j in range(n):
            print(adj[i, j], end=", ")
        print()
    
def histogram(network):
    """Return the mean histograms of trust: number of link per trust value"""

    mean = np.zeros(network.cognitive_capa+1, dtype=float)
    bins = np.arange(network.cognitive_capa+1)

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
