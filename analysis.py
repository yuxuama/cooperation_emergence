"""analysis.py

File where simulation results are analysed using different tools
"""
from utils import parse_parameters, histogram
from operation import OperationStack

in_dir = r"./out/group_of_5/"

if __name__ == '__main__':
    parameters = parse_parameters(in_dir + "parameters.yaml")
    oper = OperationStack(in_dir)

    final_trust, _ = oper.resolve(-1)
    histogram(final_trust, parameters)
