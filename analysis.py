"""analysis.py

File where tools for analysing simulation are defined
"""

import numpy as np
from utils import get_vertex_distribution
from tqdm import tqdm
from scipy.optimize import newton
from numba import njit 
from numba.typed.typedlist import List
from scipy.optimize import curve_fit

######################################################################################
# Trust distribution histogram
######################################################################################


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

######################################################################################
# Unified global measuring
######################################################################################

def measure_global(quantity, trust_adjacency_matrix, link_adjacency_matrix, capacity_table, parameters, random=False,**rand_kwargs):
    """Unified method for measurement
    Quantity among: "Asymmetry", "Individual asymmetry", "Saturation rate", "Number of link"
    WARNING: randomized only work for measures on **links**"""

    possible_quantities = {
        "Asymmetry": {
            "func": measure_link_asymmetry,
            "param": [link_adjacency_matrix]
            },
        "Individual asymmetry": {
            "func": measure_individual_asymmetry,
            "param": [link_adjacency_matrix]
            },
        "Saturation rate": {
            "func": measure_saturation_rate,
            "param": [trust_adjacency_matrix, capacity_table, parameters]
            },
        "Number of link": {
            "func": measure_number_of_link,
            "param": [link_adjacency_matrix]
            }
    }

    measure_tool = possible_quantities[quantity]

    if random:
        return randomizer(measure_tool["func"], *measure_tool["param"], **rand_kwargs)
    
    return measure_tool["func"](*measure_tool["param"])
    
def randomizer(func, *fargs, niter=300, mode="i&&o", mc_iter=100, **fkwargs):
    """Randomizer decorator for measures"""
    data = np.zeros(niter)
    fargs = list(fargs)
    initial = fargs.pop(0)

    def neutral_embedder(arg):
        return arg
    embedder = neutral_embedder
    if mode == 'i&&o':
        embedder = tqdm

    for i in embedder(range(niter)):
        randomized_initial = compute_randomized(initial, mode=mode, mc_iter=mc_iter)
        data[i] = func(randomized_initial, *fargs, **fkwargs)
    
    return np.median(data)

def measure_link_asymmetry(link_adjacency_matrix):
    """Return the proportion of link that are asymmetrical globally"""
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

def measure_individual_asymmetry(link_adjacency_matrix, full=False):
    """Return the proportion of link that are asymmetrical per vertex"""
    n = link_adjacency_matrix.shape[0]
    number_of_asymmetric = np.zeros(n)
    for i in range(n):
        for j in range(i+1, n):
            if link_adjacency_matrix[i, j] != link_adjacency_matrix[j, i]:
                    number_of_asymmetric[i] += 1
                    number_of_asymmetric[j] += 1
    number_of_asymmetric = number_of_asymmetric / n
    if full:
        return number_of_asymmetric
    return np.median(number_of_asymmetric)

def measure_saturation_rate(trust_adjacency_matrix, capacity_table, parameters):
    """Return the proportion of edje that are saturated"""
    count = 0
    total = trust_adjacency_matrix.shape[0]

    for i in range(total):
        sumload = np.sum(trust_adjacency_matrix[i])
        if sumload >= capacity_table[i]:
            count += 1

    return count / total

def measure_number_of_link(link_adjacency_matrix):
    """Return the average and sigma of the distribution of number of out links per agent"""
    number_out_links = np.sum(link_adjacency_matrix, axis=1)
    return np.mean(number_out_links), np.std(number_out_links, ddof=number_out_links.size - 1)

######################################################################################
# Xhi and etas
######################################################################################

def measure_etas_from_xhi(i, trust_adjacency_matrix, link_min):
    """Measure the value of the eta parameters for agent `i` in the network
    Return: eta, t and xhi(t)"""
    xhi = compute_xhi(i, trust_adjacency_matrix, link_min)
    if len(xhi) < 2:
        return
    size = xhi.size
    t_norm = np.arange(size) / (size - 1)
    model = lambda j, eta: (np.exp(eta * j) - 1) / (np.exp(eta) - 1)
    popt, _ = curve_fit(model, t_norm, xhi)
    return popt[0]
    
def compute_xhi(i, trust_adjacency_matrix, link_min):
    """Compute the xhi as defined in https://doi.org/10.1038/s41598-022-06066-1 for agent `i`"""
    data = trust_adjacency_matrix[i]
    if np.sum(data) == 0:
        return
    bins = np.arange(0, np.max(data)+2)
    expectations, _ = np.histogram(data, bins=bins, density=False)
    expectations = expectations[int(link_min)::]
    L = np.sum(expectations)
    xhi = expectations / L
    xhi = np.flip(xhi)
    for i in range(1, len(xhi)):
        xhi[i] += xhi[i-1]
    
    return xhi

def compute_xhi_mean(trust_adjacency_matrix, link_min_table, parameters, ph_selector="Global"):
    """Compute the xhi as defined in https://doi.org/10.1038/s41598-022-06066-1
    for the group distribution labeled by `ph_selector`"""
    if ph_selector == "Global":
        link_min = np.min(link_min_table)
    else:
        link_min = np.max(link_min_table)
        vertex_distrib = get_vertex_distribution(parameters)
        for i in range(len(vertex_distrib)):
            if vertex_distrib[i] == ph_selector:
                if link_min_table[i] < link_min:
                    link_min = link_min_table[i]
    expectations = histogram(trust_adjacency_matrix, parameters)
    expectations = expectations[ph_selector][int(link_min)::]
    L = np.sum(expectations)
    xhi = expectations / L
    xhi = np.flip(xhi)
    for i in range(1, len(xhi)):
        xhi[i] += xhi[i-1]
    
    return xhi

def compute_all_xhi_mean(trust_adjacency_matrix, link_min_table, parameters):
    """Compute the xhi function per phenotype"""
    all_xhi = histogram(trust_adjacency_matrix, parameters)
    all_link_min = {"Global": np.min(link_min_table)}
    vertex_distrib = get_vertex_distribution(parameters)
    for i in range(len(vertex_distrib)):
        if vertex_distrib[i] in all_link_min:
            all_link_min[vertex_distrib[i]] = min(all_link_min[vertex_distrib[i]], link_min_table[i])
        else:
            all_link_min[vertex_distrib[i]] = link_min_table[i]
    for key in all_xhi.keys():
        data = all_xhi[key][int(all_link_min[key])::]
        L = np.sum(data)
        xhi = data / L
        xhi = np.flip(xhi)
        for i in range(1, len(xhi)):
            xhi[i] += xhi[i-1]
        all_xhi[key] = xhi
    
    return all_xhi

def estimate_etas_with_L(trust_adjacency_matrix, link_adjacency_matrix, link_min_table):
    """Estimate the eta parameter defined in https://doi.org/10.1038/s41598-022-06066-1
    with the estimator derived from bayesian inference described in the paper"""
    n = trust_adjacency_matrix.shape[0]
    etas = np.zeros(n)

    func_prime = lambda eta: (1/(eta**2)) - (np.exp(eta)/(np.exp(eta) -1)**2)

    for i in range(n):
        L = np.sum(trust_adjacency_matrix[i] > link_min_table[i])

        selector = link_adjacency_matrix[i] > 0
        trust_values = trust_adjacency_matrix[i][selector]
        freq, bins = np.histogram(trust_values, bins=np.arange(np.min(trust_values), np.max(trust_values)+2))
        L1 = np.sum(freq * (np.flip(bins[0:bins.size-1]) - link_min_table[i])) / (np.max(trust_adjacency_matrix) - link_min_table[i])

        if L1/L > 1:
            print("WARNING: value of ratio above 1... skipping")
            continue
        if L1/L < 0:
            print("WARNING: value of ratio below 0... skipping")
            continue

        func = lambda eta: (np.exp(eta) / (np.exp(eta) - 1)) - (1/eta) - L1/L

        etas[i] = newton(func, 0.5, func_prime)
    
    return etas

def compute_xhi_ratio(i, trust_adjacency_matrix, link_min):
    """Compute the xhi ratio as defined in https://doi.org/10.1038/s41598-022-06066-1 for agent `i`"""
    data = trust_adjacency_matrix[i]
    bins = np.arange(0, np.max(data)+2)
    expectations, _ = np.histogram(data, bins=bins, density=False)
    expectations = expectations[link_min::]
    L = np.sum(expectations)
    xhi = expectations / L
    xhi = np.flip(xhi) # the scale goes from high cost to low cost
    n = xhi.size
    for i in range(1, n):
        xhi[i] += xhi[i-1]
    
    xhi_prime = np.zeros(n)
    for i in range(1, n-1):
        xhi_prime[i] = (xhi[i+1] - xhi[i-1]) / 2
    xhi_prime[0] = xhi[1] - xhi[0]
    xhi_prime[n - 1] = xhi_prime[n - 1] - xhi_prime[n - 2]

    return xhi_prime / xhi

######################################################################################
# Randomization
######################################################################################

def compute_randomized(link_adjacency_matrix, mode, mc_iter=10):
    """Return a randomised version of the network respecting certain conditions regarding the mode chose:
    Modes:
    - `'i'`: "in" respect the in degree of each node
    - `'o'`: "out" respect the out degree of each node
    - `'i&&o'`: "in" and "out" respect the composite degree (both in and out degrees)
    - `'n-link'`: just reaffect links keeping its number constant
    - `"pn-link": reaffect links but keeping the number of total links and the proportion of reciprocal link constant`"""
    n = link_adjacency_matrix.shape[0]
    new_link_adjacency = np.zeros(link_adjacency_matrix.shape, dtype=int)

    if mode == "i&&o":
        return monte_carlo_randomisation(mc_iter, link_adjacency_matrix)
    elif mode == "n-link":
        return nlink_preserving_randomisation(link_adjacency_matrix)
    elif mode == "pn-link":
        return pnlink_preserving_randomisation(link_adjacency_matrix)
    elif mode == "i":
        return one_degree_preserving_randomisation(link_adjacency_matrix, "in")
    elif mode == "o":
        return one_degree_preserving_randomisation(link_adjacency_matrix, "out")
    else:
        raise KeyError("Unknown mode for randomisation. It must be in: 'o', 'i', 'i&&o', 'n-link', 'pn-link'")

def one_degree_preserving_randomisation(link_adjacency_matrix, degree):
    """Return a degree preserving randomisation of the network:
    Possible values for `degree`:
    - "in": Preserve in-degree
    - "out": Preserve out-degree"""
    n = link_adjacency_matrix.shape[0]
    new_link_adjacency = np.zeros(link_adjacency_matrix.shape, dtype=int)
    in_degree = np.sum(link_adjacency_matrix, axis=0)
    out_degree = np.sum(link_adjacency_matrix, axis=1)
    for i in range(n):
        if degree == "in":
            draw = np.random.choice(n-1, in_degree[i], replace=False) # n-1 because a link with oneself is not allowed
            for j in draw:
                if j < i:
                    new_link_adjacency[j, i] = 1
                else:
                    new_link_adjacency[j+1, i] = 1
        elif degree == "out":
            draw = np.random.choice(n-1, out_degree[i], replace=False)
            for j in draw:
                if j < i:
                    new_link_adjacency[i, j] = 1
                else:
                    new_link_adjacency[i, j+1] = 1
        else:
            raise KeyError("Unknown mode for one-degree preserving randomisation")
    return new_link_adjacency

def nlink_preserving_randomisation(link_adjacency_matrix):
    """Return a randomised version of the network with the same number of edges"""
    n = link_adjacency_matrix.shape[0]
    new_link_adjacency = np.zeros(link_adjacency_matrix.shape, dtype=int)
    total_link_number = int(np.sum(link_adjacency_matrix))
    try:
        draw_link = np.random.choice(n*(n-1), size=total_link_number, replace=False)
    except:
        raise Exception("Can't draw random links")
    draw_link.sort()
    for index in draw_link:
        i = index // (n-1)
        j_red = index % (n-1)
        if j_red >= i:
            j_red += 1
        new_link_adjacency[i, j_red] = 1
    return new_link_adjacency

def pnlink_preserving_randomisation(link_adjacency_matrix):
    """Return a randomised version of the network with the same number of edges
    and with the same proportion of reciprocal link"""
    n = link_adjacency_matrix.shape[0]
    new_link_adjacency = np.zeros(link_adjacency_matrix.shape, dtype=int)
    total_reciprocal_link = int(np.trace(link_adjacency_matrix @ link_adjacency_matrix)) // 2
    min_unidirectional_link = int(np.sum(link_adjacency_matrix)) - total_reciprocal_link
    link = []
    # Create only unidirectional link
    draw_unidirectionals = np.random.choice(n * (n-1) // 2, size=min_unidirectional_link, replace=False)
    for t in range(min_unidirectional_link):
        index = draw_unidirectionals[t]
        p = n-1
        i = 0
        j = index % p + 1
        while index // p > 0:
            i += 1
            index = index - p
            p -= 1
            j = i+1+index
        if np.random.random(1) > 0.5:
            drawn_link = (i, j)
        else:
            drawn_link = (j, i)
        new_link_adjacency[drawn_link[0], drawn_link[1]] = 1
        link.append(drawn_link)

    # Create reciprocal link using existing unidirectional links
    draw_reciprocal = np.random.choice(len(link), size=total_reciprocal_link, replace=False)
    for index in draw_reciprocal:
        drawn_link = link[index]
        new_link_adjacency[drawn_link[1], drawn_link[0]] = 1

    return new_link_adjacency
    
@njit
def monte_carlo_randomisation(niter, link_adjacency):
    """Randomise the network preserving both in and out degree"""

    new_link_adjacency = link_adjacency.copy()
    swappable_link = compute_swappable_links(new_link_adjacency)
    
    for it in range(niter):
        if len(swappable_link) == 0:
            print("ERROR: no more swappable link")
            print("Breaking loop at iteration: ", it)
            break

        draw = np.random.randint(len(swappable_link))
        links = swappable_link[draw]
        new_link_adjacency[links[2], links[1]] = new_link_adjacency[links[0], links[1]]
        new_link_adjacency[links[0], links[1]] = 0
        new_link_adjacency[links[0], links[3]] = new_link_adjacency[links[2], links[3]]
        new_link_adjacency[links[2], links[3]] = 0

        # Remove old edges
        i = 0
        while i < len(swappable_link): 
            test_links = swappable_link[i]
            if link_invalidate_test_link(test_links, links):
                swappable_link.pop(i) # Because pop is len dependent
            else:
                i += 1
        
        # Find new swappable
        swappable_links_from_link(links[2], links[1], new_link_adjacency, swappable_link)
        swappable_links_from_link(links[0], links[3], new_link_adjacency, swappable_link, other_link=(links[2], links[1]))
    
    return new_link_adjacency

@njit
def compute_swappable_links(link_adjacency):
    """Return all possible swappable link of the matrix conserving its structure"""
    n = link_adjacency.shape[0]
    swappable_link = List()
    new_link_adjacency = link_adjacency.copy()
    indexes = np.arange(n)
    for i in range(n):
        for j in range(i+1, n):
            if new_link_adjacency[i, j] > 0:
                c_selector = new_link_adjacency[i] < 1
                l_selector = new_link_adjacency[::,j] < 1
                for k in range(i+1):
                    l_selector[k] = False
                c_selector[i] = False
                l_selector[j] = False
                c_index = indexes[c_selector]
                l_index = indexes[l_selector]

                if len(l_index) == 0 or len(c_index) == 0:
                    continue

                for k in range(len(l_index)):
                    for l in range(len(c_index)):
                        if link_adjacency[l_index[k], c_index[l]] > 0:
                            swappable_link.append((i, j, l_index[k], c_index[l]))

    return swappable_link

@njit
def swappable_links_from_link(i, j, link_adjacency, swappable_link, other_link=(-1, -1)):
    """Add to `swappable_link` the swappable link containing the link i -> j in `link_adjacency`
    the parameter `other_link` is here to prevent doble link in swappable_link"""
    n = link_adjacency.shape[0]
    c_selector = link_adjacency[i] < 1
    l_selector = link_adjacency[::,j] < 1
    c_selector[i] = False
    l_selector[j] = False
    indexes = np.arange(n)
    c_index = indexes[c_selector]
    l_index = indexes[l_selector]
    
    if len(c_index) == 0 or len(l_index) == 0:
        return
    
    for k in range(len(l_index)):
        for l in range(len(c_index)):
            if l_index[k] == other_link[0] and c_index[l] == other_link[1]:
                continue
            elif link_adjacency[l_index[k], c_index[l]] > 0:
                swappable_link.append((i, j, l_index[k], c_index[l]))

@njit
def link_invalidate_test_link(test_links, links):
    """Return True if one of the change of link `links` invalidates `test_links`"""
    # Test for link that disappeared
    if links[0] == test_links[0] and links[1] == test_links[1]:
        return True
    elif links[0] == test_links[2] and links[1] == test_links[3]:
        return True
    elif links[2] == test_links[0] and links[3] == test_links[1]:
        return True
    elif links[2] == test_links[2] and links[3] == test_links[3]:
        return True
    # test for invalid XSWAP in-out square
    if links[2] == test_links[0] and links[1] == test_links[3]:
        return True
    elif links[2] == test_links[2] and links[1] == test_links[1]:
        return True
    if links[0] == test_links[0] and links[3] == test_links[3]:
        return True
    elif links[0] == test_links[2] and links[3] == test_links[1]:
        return True

    return False
    
def poisson(k, lamb):
    return np.power(lamb, k) * np.exp(-lamb) / factorial(k)

def factorial(k):
    if k == 0:
        return 1
    return k * factorial(k-1)

if __name__ == "__main__":
    test = np.random.randint(2, size=36).reshape((6, 6))
    for i in range(test.shape[0]):
        test[i, i] = 0
    number_reciprocal = int(np.trace(test @ test)) // 2
    print(test)
    print(number_reciprocal)
    print(np.sum(test))
    net_rand = compute_randomized(test, "pn-link")
    print(net_rand)
    print(int(np.trace(net_rand @ net_rand)) // 2)
    print(np.sum(net_rand))
    #print(in_degrees == np.sum(net_rand, axis=0))
    #print(out_degrees == np.sum(net_rand, axis=1))
    #print(measure("Individual asymmetry", net_rand, net_rand, 0, niter=100, mode="i&&o", mc_iter=10))