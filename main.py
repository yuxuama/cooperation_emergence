"""main.py
Execute simulation of the interacting network
"""
from graph import Network
from utils import parse_parameters
import numpy as np

parameters_file = r"./parameters.yaml"

if __name__ == '__main__':
    parameters = parse_parameters(parameters_file)
    net = Network(parameters)
    for v in net.verteces:
        print(v)
    net.create_link(1, 2)
    adjency_matrix = np.zeros((10, 10), dtype=bool)
    adjency_matrix[2, 1] = True
    net.set_link_from_adjency_matrix(adjency_matrix)
    print(net.link)