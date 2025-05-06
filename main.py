"""main.py
Execute simulation of the interacting network
"""
from graph import Network
from utils import parse_parameters

parameters_file = r"./parameters.yaml"

if __name__ == '__main__':
    parameters = parse_parameters(parameters_file)
    net = Network(parameters)
    net.play()