""" utils.py
All utils functions 
"""

from yaml import safe_load
import numpy as np

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

if __name__ == "__main__":
    print(parse_parameters(r"parameters.yaml"))
