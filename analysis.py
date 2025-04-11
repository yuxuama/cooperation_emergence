"""analysis.py

File where tools for analysing simulation are defined
"""

import numpy as np
from utils import get_vertex_distribution


def histogram(trust_adjacency_matrix, parameters, bins=None):
    """Return histogram of the weight distribution for each phenotype and the average distribution in log scale"""
    size = parameters["Community size"]
    if bins is None:
        maxi = int(np.ceil(np.max(trust_adjacency_matrix)))+1
    else:
        maxi = bins.size-1
    phenotype_table = get_vertex_distribution(parameters)

    # Generating each histogram
    phenotype_mean = {"Global": np.zeros(maxi, dtype=float)}
    phenotype_count = {"Global": size}

    for i in range(size):
        data = trust_adjacency_matrix[i]
        n, _ = np.histogram(data, bins=np.arange(int(np.ceil(np.max(data)))+2), density=False)
        n[0] -= 1 # Il faut enlever le fait que la personne n'a pas de lien avec elle-même
        ph = phenotype_table[i]
        if not ph in phenotype_mean:
            phenotype_mean[ph] = np.zeros(maxi, dtype=float)
        if not ph in phenotype_count:
            phenotype_count[ph] = 0
        phenotype_count[ph] += 1
        for i in range(len(n)):
            phenotype_mean[ph][i] += n[i]
            phenotype_mean["Global"][i] += n[i]
    
    for key in phenotype_mean.keys():
        phenotype_mean[key] /= phenotype_count[key] 

    return phenotype_mean

def measure_link_asymmetry(link_adjacency_matrix):
    """Return the proportion of link that are asymmetrical"""
    size = link_adjacency_matrix.shape[0]
    number_of_link = 0
    number_of_asymmetric_link = 0

    for i in range(size):
        for j in range(i+1, size):
            if link_adjacency_matrix[i, j] != link_adjacency_matrix[j, i]:
                number_of_asymmetric_link += 1
                number_of_link += 1
            elif link_adjacency_matrix[i, j] > 0:
                number_of_link += 1
    if number_of_link == 0:
        return 0
    else:
        return number_of_asymmetric_link / number_of_link

def measure_individual_asummetry_rate(link_adjacency_matrix):
    """Return the proportion of link that are asymmetrical per vertex"""
    n = link_adjacency_matrix.shape[0]
    number_of_link = np.sum(link_adjacency_matrix, axis=1)
    number_of_asymmetric = np.zeros(n)

    for i in range(n):
        for j in range(i+1, n):
            if link_adjacency_matrix[i, j] != link_adjacency_matrix[j, i]:
                if link_adjacency_matrix[i, j] > 0:
                    number_of_asymmetric[i] += 1
                else:
                    number_of_asymmetric[j] += 1
    for i in range(n):
        if number_of_link[i] == 0:
            number_of_asymmetric[i] = 0
        else:
            number_of_asymmetric[i] /= number_of_link[i]
    return number_of_asymmetric


def measure_saturation_rate(trust_adjacency_matrix, max_load):
    """Return the proportion of edje that are saturated"""
    count = 0
    total = trust_adjacency_matrix.shape[0]

    for i in range(total):
        sumload = np.sum(trust_adjacency_matrix[i])
        if sumload == max_load:
            count += 1

    return count / total

def poisson(k, lamb):
    return np.power(lamb, k) * np.exp(-lamb) / factorial(k)

def factorial(k):
    if k == 0:
        return 1
    return k * factorial(k-1)

