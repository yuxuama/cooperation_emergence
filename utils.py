""" utils.py
All utils functions 
"""

from yaml import safe_load, dump
import numpy as np
import h5py
import os

"""UTILS"""

def parse_parameters(yaml_file):
    """Load parameters for the simulation from a yaml file"""
    stream = open(yaml_file, 'r')
    return safe_load(stream)

def save_parameters(parameters, dir_path, prefix=""):
    """Save the parameters represented by `data` in the `dir_path` directory"""
    stream = open(dir_path + prefix + "parameters.yaml", 'w')
    dump(parameters, stream=stream, default_flow_style=False)

def save_parameters_in_hdf5(parameters, hdf5_file: h5py.File):
    """Save the parameters `parameters` in the HDF5 file `hdf5_file` (must be of type h5py.File)"""
    pgroup = hdf5_file.create_group("Parameters", track_order=True)
    for key, value in parameters.items():
        if key != "Strategy distributions":
            pgroup[key] = value
    strat_subgroup = pgroup.create_group("Strategy distributions")
    for key, value in parameters["Strategy distributions"].items():
        strat_subgroup[key] = value

def extract_parameters_from_hdf5(filepath):
    """Extract the parameters from the HDF5 file at `filepath`"""
    parameters = {}
    string_keys = {"Heuristic", "Init", "Output directory", "Save mode"}
    hdf5_file = h5py.File(filepath)
    grp = hdf5_file["Parameters"]
    for p in grp.keys():
        if p != "Strategy distributions":
            if p in string_keys:
                parameters[p] = grp.get(p)[()].decode("ascii")
            else:
                parameters[p] = grp.get(p)[()]
    subgrp = grp["Strategy distributions"]
    strategy_distrib = {}
    for ph in subgrp.keys():
        strategy_distrib[ph] = subgrp.get(ph)[()]
    parameters["Strategy distributions"] = strategy_distrib

    return parameters

def print_parameters(parameters):
    """Print the parameters extracted from a yaml file"""
    for key, values in parameters.items():
        print("..", key, ": ", values)

def generate_output_name_from_parameters(parameters, postfix):
    """Generate a name that represent uniquely this network
    format: '<Heurisitc>_<Distribution>_I<Init>_L<Link minimum>_C<Cognitive capacity>_S<Size>_T<Temperature>_<postfix>
    distribution format: "Envious: 0.3" -> E3"""
    name = ""
    name += parameters["Heuristic"]
    name += "_"
    # distribution
    distribution = ""
    keys = list(parameters["Strategy distributions"].keys())
    keys = sorted(keys)
    for key in keys:
        prop = str(parameters["Strategy distributions"][key])
        if len(prop) < 2:
            distribution += key[0] + prop
        else:
            distribution += key[0] + prop[2::]
    name += distribution
    
    # other
    name += "_"
    if "Number of groups" in parameters and parameters["Init"] == "Groups":
        name += str(parameters["Number of groups"])
    name += parameters["Init"]
    name += "_"
    name += "L" + str(parameters["Link minimum"]) + "_"
    name += "C" + str(parameters["Cognitive capacity"]) + "_"
    name += "S" + str(parameters["Community size"]) + "_"
    name += "T" + str(parameters["Temperature"])

    if postfix == "":
        return name
    
    return name + "_" + str(postfix)

def model_parameters_change(old_parameters, parameters):
    """Compare the model parameters 'Link minimum', 'Cognitive capacity' and 'Temperature'"""

    if old_parameters["Save mode"] != parameters["Save mode"]:
        raise KeyError("Reload can be done only with the same saving mode parameters")
    if old_parameters["Community size"] != parameters["Community size"]:
        raise KeyError("Incompatible sizes of the networks: cannot reload")
    if old_parameters["Strategy distributions"] != parameters["Strategy distributions"]:
        raise KeyError("Incompatible phenotype distribution: cannot reload")
    if old_parameters["Link minimum"] != old_parameters["Link minimum"]:
        raise KeyError("Incompatible phenotype distribution: cannot reload")

    model_p_name = ["Cognitive capacity", "Temperature", "Heuristic"]
    model_old_p = []
    model_p = []
    diff = np.zeros(3, dtype=bool)
    for i in range(len(model_p_name)):
        old = old_parameters[model_p_name[i]]
        new = parameters[model_p_name[i]]
        if old != new:
            diff[i] = True
        model_old_p.append(old)
        model_p.append(new)
    
    if np.sum(diff) == 0:
        return
    
    print("WARNING: some parameters of the simulation changed:")
    for i in range(len(diff)):
        if diff[i]:
            print("{0}: {1} --> {2}".format(model_p_name[i], model_old_p[i], model_p[i]))
    print()
    proceed = input("Proceed ? [y]/n ")
    if proceed == "" or proceed == "y":
        return
    else:
        print("Aborting reload sequence")
        exit(1)


def readable_adjacency(adjacency_matrix=np.ndarray):
    """Gives the adjacency matric in a form usalble in https://graphonline.top/en/create_graph_by_matrix"""
    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            print(adjacency_matrix[i, j], end=", ")
        print()

def get_vertex_distribution(parameters):
    """Give the link vertex index - vertex phenotype according to the parameters"""
    strategy_distrib = parameters["Strategy distributions"]
    size = parameters["Community size"]
    phenotypes = list(strategy_distrib.keys())
    
    distribution_grid = [0] # Used to initialized populations of each phenotypes according to parameters.
    for p in strategy_distrib.values():
        distribution_grid.append(distribution_grid[-1] + p)
    distribution_grid = size * np.array(distribution_grid)

    pointer = 0
    table = ["" for _ in range(size)]
    for i in range(size):
        while i > distribution_grid[pointer]:
            pointer += 1
        table[i] = phenotypes[pointer-1]
    
    return table

def load_hdf5(filename):
    """Return the trust and the link matrices stored in the hdf5 file `filename`"""
    f = h5py.File(filename, 'r')
    link = np.array(f.get("Link"), dtype=int)
    trust = np.array(f.get("Trust"))
    return trust, link

def list_all_hdf5(dirpath):
    """Return the list of all the hdf5 files in the directory with path `dirpath`"""
    files = [dirpath + f for f in os.listdir(dirpath)]
    h5_files = []
    for i in range(len(files)):
        if files[i].endswith(".h5"):
            h5_files.append(files[i])
    return h5_files

def proceed(text):
    """Ask if want to proceed"""
    print("Warning: ", text)
    print()
    proceed = input("Proceed ? [y]/n ")
    if proceed == "y" or proceed == "":
        return True
    print("Aborting process")
    return False

"""INIT functions"""

def create_social_group(size, assignation, adjacency_matrix, min_trust):
    """Create a fully connected graph of size `size` among an unassigned group of people
    Allow to create disconnected social group"""
    remaining = assignation.size - np.sum(assignation)
    
    sample = np.random.choice(remaining, size, replace=False)
    sample = np.sort(sample)
    real_sample = np.zeros(size, dtype=int)

    count = 0
    s_pointer = 0
    for i in range(assignation.size):
        if not assignation[i]:
            if sample[s_pointer] == count:
                real_sample[s_pointer] = i
                s_pointer += 1
                if s_pointer >= size:
                    break
            count += 1
    for i in range(size):
        assignation[real_sample[i]] = True
        for j in range(size):
            if j != i:
                adjacency_matrix[real_sample[i], real_sample[j]] = min_trust

def create_random_init(adjacency_matrix, cognitive_capa, min_trust):
    """Create a network with `n` random trust link"""
    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                s = np.sum(adjacency_matrix[i])
                remain = cognitive_capa - s
                if remain <= 0:
                    adjacency_matrix[i, j] = 0
                else:
                    adjacency_matrix[i, j] = np.random.randint(0, min(2 * min_trust, remain+1))

