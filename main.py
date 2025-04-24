"""main.py
Execute simulation of the interacting network
"""
from graph import Network
from utils import parse_parameters, create_social_group, create_random_init
import numpy as np

parameters_file = r"./parameters.yaml"

if __name__ == '__main__':
    parameters = parse_parameters(parameters_file)
    net = Network(parameters)
    
    size = net.size
    min_trust = net.min_trust
    cog_capa = net.cognitive_capa

    adjacency_matrix = np.zeros((size, size))
    assignation = np.zeros(size, dtype=bool)

    #for _ in range(5):
    #    create_social_group(13, assignation, adjacency_matrix, min_trust)

    #create_random_init(adjacency_matrix, cog_capa, min_trust)

    net.set_adjacency_trust_matrix(adjacency_matrix)
    net.play()