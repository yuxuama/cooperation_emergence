"""main.py
Execute simulation of the interacting network
"""
from graph import Network
from utils import parse_parameters, create_social_groupe, readable_adjacency
import numpy as np

parameters_file = r"./parameters.yaml"
out = r"./out/test/"

if __name__ == '__main__':
    parameters = parse_parameters(parameters_file)
    net = Network(parameters, out)
    
    size = net.size
    min_trust = net.min_trust

    adjacency_matrix = np.zeros((size, size))
    assignation = np.zeros(size, dtype=bool)

    for _ in range(2):
        create_social_groupe(3, assignation, adjacency_matrix, min_trust)

    net.set_adjacency_trust_matrix(adjacency_matrix)
    readable_adjacency(net.get_adjacency_link_matrix())
    print()
    net.play()
    readable_adjacency(net.get_adjacency_link_matrix())

    