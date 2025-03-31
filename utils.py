""" utils.py
All utils functions 
"""

from yaml import safe_load

def parse_parameters(yaml_file):
    """Load parameters for the simulation from a yaml file"""
    stream = open(yaml_file, 'r')
    return safe_load(stream)

if __name__ == "__main__":
    print(parse_parameters(r"parameters.yaml"))
