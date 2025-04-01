"""graph.py
Implement the mathematical graph structure of the network
"""
import numpy as np

"""Graph class for network structure"""

class Network:

    def __init__(self, parameters):
        ## defining properties
        self.max_iter = parameters["Number of interaction"]
        self.size = parameters["Community size"]
        self.min_link = parameters["Community minimum"]
        self.temp = parameters["Temperature"]
        self.cognitive_capa = parameters["Cognitive capacity"]
        
        strategy_distrib = parameters["Strategy distributions"]
        self.phenotypes = list(strategy_distrib.keys())
        self.distribution_grid = [0] # Used to initialized populations of each phenotypes according to parameters.
        for p in strategy_distrib.values():
            self.distribution_grid.append(self.distribution_grid[-1] + p)
        self.distribution_grid = self.size * np.array(self.distribution_grid) 
        
        ## creating structure
        # Creating verteces

        self.verteces = np.zeros(self.size, dtype=Vertex)
        for i in range(self.size):
            self.create_vertex(i)
        
        # Creating trust adjency matrix
        self.trust = np.zeros((self.size, self.size))

        # Create link matrix
        self.link = [set() for _ in range(self.size)]
    
    def create_vertex(self, index):
        """Add a person into the simulation"""
        # Choosing phenotypes according to distribution
        p = 0
        while index >= self.distribution_grid[p]:
            p += 1
        self.verteces[index] = Vertex(self.phenotypes[p-1], index)

    def create_link(self, start, end):
        """Create a link between two vertices"""
        self.link[start].add(end)
    
    def remove_link(self, start, end):
        """Remove a link between two vertices"""
        self.link[start].remove(end)

    def interact(self):
        """Choose randomly two vertices and a game matrix a resolve the interaction
        We follow the paper and chose a game matrix of the form
        | R | S |
        | T | P |
        where R = 10, P = 5
        and S \in [0, 10]
        and T \in [5, 15]
        """
        
        pair = np.random.choice(self.size, 2)
        T = np.random.randint(5, 15)
        S = np.random.randint(0, 10)

        game_matrix = np.array([[10, S], [T, 5]])
        choices = np.zeros(2)
        choices[0] = self.verteces[pair[0]].choose(game_matrix, self.temp)
        choices[1] = self.verteces[pair[1]].choose(game_matrix, self.temp)

        # ... To be continued
    
    def get_adjency_link_matrix(self):
        """Return the adjency matrix of all link"""
        link_adjency_matrix = np.zeros((self.size, self.size), dtype=bool)
        for i in range(self.size):
            for j in self.link[i]:
                link_adjency_matrix[i, j] = True

        return link_adjency_matrix

"""Vertex class for handling people"""

class Vertex:

    def __init__(self, phenotype, index):
        self.phenotype = phenotype
        self.index = index
        self.load = 0

    def choose(self, game_matrix, temperature):
        """Return the choice of the person in the dyadic game based on temperature and phenotype
        0 -> cooperates
        1 -> defects
        """
        T = game_matrix[1, 0]
        S = game_matrix[0, 1]
        max_gain = max(game_matrix[0, 0], abs(T-S))

        strategic_response = 1
        if self.phenotype == "Envious":
            if S - T >= 0:
                strategic_response = 0
        elif self.phenotype == "Pessimist":
            if S > game_matrix[1, 1]:
                strategic_response = 0 
        elif self.phenotype == "Optimist":
            if T < game_matrix[0, 0]:
                strategic_response = 0
        elif self.phenotype == "Trustful":
            strategic_response = 0
        elif self.phenotype == "Random":
            draw = np.random.rand
            if draw >= 0.5:
                strategic_response = 0

        if temperature == 0:
            return strategic_response
        
        draw = np.random.rand()
        if draw > np.exp(- max_gain / temperature):
            return strategic_response

        return 1 - strategic_response 
    
    def __str__(self):
        """Allows to print vertex in a readable way"""
        return self.phenotype + " " + str(self.index)