"""main.py
Execute simulation of the interacting network
"""
from graph import Network
from utils import parse_parameters
import numpy as np
from dataset import measure_triadic_pattern_phenotype_combination
from plot import plot_phenotype_combination_per_triangle
from tqdm import tqdm
import matplotlib.pyplot as plt 

parameters_file = r"./parameters.yaml"

if __name__ == '__main__':
    parameters = parse_parameters(parameters_file)
    net = Network()
    net.init_with_parameters(parameters)
    net.play()
