"""analysis.py

File where tools for analysing simulation are defined
"""

import numpy as np
from utils import get_vertex_distribution
from tqdm import tqdm


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

def measure_individual_asymmetry(link_adjacency_matrix):
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
    return np.median(number_of_asymmetric)

def measure_random_individual_asymmetry(link_adjacency_matrix, niter, mode="i&&o", mc_iter=1000):
    """Return the median asymmetry rate of a randomized network"""

    rates = np.zeros(niter)
    if mode == "i&&o":
        for i in tqdm(range(niter)):
            randomized_network = randomized(link_adjacency_matrix, mode=mode, mc_iter=mc_iter)
            rates[i] = measure_individual_asymmetry(randomized_network)
    else:
        for i in range(niter):
            randomized_network = randomized(link_adjacency_matrix, mode=mode, mc_iter=mc_iter)
            rates[i] = measure_individual_asymmetry(randomized_network)
    return np.median(rates)

def measure_saturation_rate(trust_adjacency_matrix, max_load):
    """Return the proportion of edje that are saturated"""
    count = 0
    total = trust_adjacency_matrix.shape[0]

    for i in range(total):
        sumload = np.sum(trust_adjacency_matrix[i])
        if sumload == max_load:
            count += 1

    return count / total

def measure_number_of_link(link_adjacency_matric):
    """Return the average and sigma of the distribution of number of out links per agent"""
    number_out_links = np.sum(link_adjacency_matric, axis=1)
    return np.mean(number_out_links), np.std(number_out_links, ddof=number_out_links.size - 1) 

def randomized(link_adjacency_matrix, mode, mc_iter=10):
    """Return a randomised version of the network respecting certain conditions regarding the mode chose:
    Modes:
    - `'i'`: "in" respect the in degree of each node
    - `'o'`: "out" respect the out degree of each node
    - `i&&o`: "in" and "out" respect the composite degree (both in and out degrees)"""
    n = link_adjacency_matrix.shape[0]
    in_degree = np.sum(link_adjacency_matrix, axis=0)
    out_degree = np.sum(link_adjacency_matrix, axis=1)
    new_link_adjacency = np.zeros(link_adjacency_matrix.shape)

    if mode == "i&&o":
        return monte_carlo_randomisation(mc_iter, link_adjacency_matrix)
    
    for i in range(n):
        if mode == "i":
            draw = np.random.choice(n-1, in_degree[i], replace=False) # n-1 because a link with oneself is not allowed
            for j in draw:
                if j < i:
                    new_link_adjacency[j, i] = 1
                else:
                    new_link_adjacency[j+1, i] = 1
        elif mode == "o":
            draw = np.random.choice(n-1, out_degree[i], replace=False)
            for j in draw:
                if j < i:
                    new_link_adjacency[i, j] = 1
                else:
                    new_link_adjacency[i, j+1] = 1
    
    return new_link_adjacency
    

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
        new_link_adjacency[links[0], links[1]] = 0
        new_link_adjacency[links[2], links[3]] = 0
        new_link_adjacency[links[2], links[1]] += 1 
        new_link_adjacency[links[0], links[3]] += 1

        # Remove old edges
        to_pop = []
        for i in range(len(swappable_link)):
            test_links = swappable_link[i]
            if link_invalidate_test_link(test_links, links):
                to_pop.append(i-len(to_pop)) # Because pop is len dependent
        for i in to_pop:
            swappable_link.pop(i)
        
        # Find new swappable
        swappable_links_from_link(links[2], links[1], new_link_adjacency, swappable_link)
        swappable_links_from_link(links[0], links[3], new_link_adjacency, swappable_link)
    
    return new_link_adjacency

def compute_swappable_links(link_adjacency):
    """Return all possible swappable link of the matrix conserving its structure"""
    n = link_adjacency.shape[0]
    swappable_link = []
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
                l_filtered_indexes = indexes[l_selector]
                c_filtered_indexes = indexes[c_selector]
                
                selector_shape = (np.sum(l_selector), np.sum(c_selector))
                if selector_shape[0] == 0 or selector_shape[1] == 0:
                    continue
                matrix_selector = np.outer(l_selector, c_selector)
                filtered_adjacency = new_link_adjacency[matrix_selector].reshape((selector_shape[0], selector_shape[1]))
                for k in range(selector_shape[0]):
                    for l in range(selector_shape[1]):
                        if filtered_adjacency[k, l] > 0:
                            swappable_link.append([i, j, l_filtered_indexes[k], c_filtered_indexes[l]])

    return swappable_link

def swappable_links_from_link(i, j, link_adjacency, swappable_link):
    """Add to `swappable_link` the swappable link containing the link i -> j in `link_adjacency`"""
    n = link_adjacency.shape[0]
    indexes = np.arange(n)
    c_selector = link_adjacency[i] < 1
    l_selector = link_adjacency[::,j] < 1
    c_selector[i] = False
    l_selector[j] = False
    c_filtered_indexes = indexes[c_selector]
    l_filtered_indexes = indexes[l_selector]
    
    selector_shape = (np.sum(l_selector), np.sum(c_selector))
    if selector_shape[0] == 0 or selector_shape[1] == 0:
        return
    
    matrix_selector = np.outer(l_selector, c_selector)
    filtered_adjacency = link_adjacency[matrix_selector].reshape((selector_shape[0], selector_shape[1]))
    for k in range(selector_shape[0]):
        for l in range(selector_shape[1]):
            if filtered_adjacency[k, l] > 0:
                swappable_link.append([i, j, l_filtered_indexes[k], c_filtered_indexes[l]])

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
    in_degrees = np.sum(test, axis=0)
    out_degrees = np.sum(test, axis=1)
    print(test)
    net_rand = randomized(test, "i&&o", mc_iter=1000)
    print(net_rand)
    print(in_degrees == np.sum(net_rand, axis=0))
    print(out_degrees == np.sum(net_rand, axis=1))

