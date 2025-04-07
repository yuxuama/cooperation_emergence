"""main.py
Execute simulation of the interacting network
"""
from graph import Network
from utils import parse_parameters, create_social_groupe, readable_adjacency, histogram
import numpy as np

parameters_file = r"./parameters.yaml"
out = r"./out/group_of_5_random_only/"

if __name__ == '__main__':
    parameters = parse_parameters(parameters_file)
    net = Network(parameters, out)
    
    size = net.size
    min_trust = net.min_trust

    adjacency_matrix = np.zeros((size, size))
    assignation = np.zeros(size, dtype=bool)

    for _ in range(15):
        create_social_groupe(5, assignation, adjacency_matrix, min_trust)

    net.set_adjacency_trust_matrix(adjacency_matrix)
    net.play()