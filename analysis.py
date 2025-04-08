"""analysis.py

File where simulation results are analysed using different tools
"""
from utils import parse_parameters, histogram, measure_link_asymmetry, poisson, plot_histogram
from operation import OperationStack
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

in_dir = r"./out/empty_simple_poisson_capa/"

if __name__ == '__main__':
    parameters = parse_parameters(in_dir + "parameters.yaml")
    oper = OperationStack(in_dir)

    inter = len(oper)//5
    final_trust, final_link = oper.resolve(inter)
    print("Asymmetry rate: ", measure_link_asymmetry(final_link))
    ph_mean, mean = histogram(final_trust, parameters)

    t = np.arange(0, mean.size)
    model = lambda x, a, b: np.exp(a*x + b)
    popt, _ = curve_fit(model, t, mean, (-1, 4))
    print(popt)
    
    data = np.zeros(mean.size)
    n = parameters["Community size"]
    for i in range(mean.size):
        data[i] = parameters["Community size"] * poisson(i, 2 * inter / (n * (n-1)))
    fig, ax = plot_histogram(ph_mean, mean, parameters, log=True)
    #ax[0].plot(t, data)
    #ax[2].plot(t, t * popt[0] + popt[1])
    plt.show()
