"""main.py
Execute simulation of the interacting network
"""
from graph import Network
from utils import parse_parameters
import numpy as np
from dataset import measure_global_frequency_triadic_pattern
from tqdm import tqdm

parameters_file = r"./parameters.yaml"

if __name__ == '__main__':
    parameters = parse_parameters(parameters_file)
    net = Network()
    
    N = 60
    transitivity_rate = np.zeros(N)
    for i in tqdm(range(N)):
        net.reset(new_seed=i)
        net.init_with_parameters(parameters)
        net.play()

        l = net.get_link_adjacency_matrix().astype(int)
        niter = parameters["Number of interaction"]
        n = parameters["Community size"]
        number_of_triangles = n * (n-1) * (n-2) / 6
        tri_pattern_freq = measure_global_frequency_triadic_pattern(
            l,
            parameters,
            niter
        )
        dtga = tri_pattern_freq.group_by("Transitive").aggregate("Number")
        data = dtga.get_item(True).get_item("Number").get_all_item().values()
        transitivity_rate[i] = sum(data) / number_of_triangles
    print("Mean: ", np.mean(transitivity_rate))
    print("Std: ", np.std(transitivity_rate, ddof=N-1))
    for t in transitivity_rate:
        print(t)