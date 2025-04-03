"""main.py
Execute simulation of the interacting network
"""
from graph import Network
from utils import parse_parameters, readable_adjency, histogram
import numpy as np

parameters_file = r"./parameters.yaml"

if __name__ == '__main__':
    parameters = parse_parameters(parameters_file)
    net = Network(parameters)
    net.play()
    for v in net.vertices:
        print(v)
    adj = net.get_adjency_link_matrix()
    readable_adjency(adj)
    histogram(net)
    