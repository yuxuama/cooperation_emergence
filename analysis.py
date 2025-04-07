"""analysis.py

File where simulation results are analysed using different tools
"""
from utils import parse_parameters, histogram, measure_link_asymmetry
from operation import OperationStack

in_dir = r"./out/group_of_5/"

if __name__ == '__main__':
    parameters = parse_parameters(in_dir + "parameters.yaml")
    oper = OperationStack(in_dir)

    final_trust, final_link = oper.resolve(50000)
    print("Asymmetry rate: ", measure_link_asymmetry(final_link))
    histogram(final_trust, parameters)
