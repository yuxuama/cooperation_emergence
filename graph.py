"""graph.py
Implement the mathematical graph structure of the network
"""
import numpy as np
  
"""Graph class for network structure"""

class Network:

    def __init__(self, parameters):
        ## defining properties
        self.parameters = parameters
        self.max_iter = parameters["Number of interaction"]
        self.size = parameters["Community size"]
        self.min_trust = parameters["Link minimum"]
        self.temp = parameters["Temperature"]
        self.cognitive_capa = parameters["Cognitive capacity"]
        self.strategy_distrib = parameters["Strategy distributions"]

        self.phenotypes = list(self.strategy_distrib.keys())
        self.distribution_grid = [0] # Used to initialized populations of each phenotypes according to parameters.
        for p in self.strategy_distrib.values():
            self.distribution_grid.append(self.distribution_grid[-1] + p)
        self.distribution_grid = self.size * np.array(self.distribution_grid)
         
        ## creating structure
        # Creating vertices

        self.vertices = np.zeros(self.size, dtype=Vertex)
        for i in range(self.size):
            self.create_vertex(i)
    
    def create_vertex(self, index):
        """Add a person into the simulation"""
        # Choosing phenotypes according to distribution
        p = 0
        while index >= self.distribution_grid[p]:
            p += 1
        self.vertices[index] = Vertex(self.phenotypes[p-1], index, self.parameters)

    def create_link(self, start, end):
        """Create a link between two vertices
        `start`: index or Vertex to start the link with
        `end`: index or Vertex to end the link with"""
        if type(start) == int:
            start = self.vertices[start]
        if type(end) == int:
            end = self.vertices[end]
        start.create_link(end)       
    
    def remove_link(self, start, end):
        """Remove a link between two vertices
        `start`: index or Vertex to start the link with
        `end`: index or Vertex to end the link with"""
        if type(start) == int:
            start = self.vertices[start]
        if type(end) == int:
            end = self.vertices[end]
        start.remove_link(end)

    def is_linked(self, start, end):
        """return True if start and end are linked"""
        if type(start) == int:
            start = self.vertices[start]
        if type(end) == int:
            end = self.vertices[end]
        start.is_linked(end)

    def trust_value(self, start, end):
        """return the trust value between start and end
        `start`: index or Vertex to start the link with
        `end`: index or Vertex to end the link with"""
        if type(start) == int:
            start = self.vertices[start]
        if type(end) == int:
            end = self.vertices[end]
            
        return start.trust_in(end)

    def interact(self):
        """Choose randomly two vertices and a game matrix a resolve the interaction
        We follow the paper and chose a game matrix of the form
        | R | S |
        | T | P |
        where R = 10, P = 5
        and S \in [0, 10]
        and T \in [5, 15]
        """

        gain = 1
        loss = -1
        
        pair = np.random.choice(self.size, 2, replace=False)
        T = np.random.rand()*10 + 5 # Random number between 5 and 15
        S = np.random.rand()*10 # Random number between 0 and 10

        game_matrix = np.array([[10, S], [T, 5]])
        
        v1 = self.vertices[pair[0]]
        v2 = self.vertices[pair[1]]
        
        choice1, happy1 = v1.choose(v2, game_matrix, self.temp)
        choice2, happy2 = v2.choose(v1, game_matrix, self.temp)
        
        if choice2 == happy1:
            v1.update_trust(v2, gain)
        else:
            v1.update_trust(v2, loss)
        
        if choice1 == happy2:
            v2.update_trust(v1, gain)
        else:
            v2.update_trust(v1, loss)


    def play(self):
        """Run the simulation"""
        for _ in range(self.max_iter):
            self.interact()

    def get_adjency_trust_matrix(self):
        """Return the adjency matrix of all trust"""
        trust_adjency_matrix = np.zeros((self.size, self.size))
        for i in range(self.size):
            v = self.vertices[i]
            for j in range(self.size):
                trust_adjency_matrix[i, j] = v.trust_in(self.vertices[j])
        return trust_adjency_matrix

    def set_adjency_trust_matrix(self, adjency_matrix):
        """Set all trust value to correspond to the adjency matrix"""

        assert adjency_matrix.shape == (self.size, self.size)
        assert np.max(np.sum(adjency_matrix, axis=1)) <= self.cognitive_capa
        
        for i in range(self.size):
            v = self.vertices[i]
            for j in range(self.size):
                vend = self.vertices[j]
                v.update_trust(vend, adjency_matrix[i, j])

    def get_adjency_link_matrix(self):
        """Return the adjency matrix of all link"""
        link_adjency_matrix = np.zeros((self.size, self.size), dtype=bool)
        for i in range(self.size):
            v = self.vertices[i]
            for vend in v.link:
                link_adjency_matrix[i, vend.index] = True

        return link_adjency_matrix
    
    def set_link_from_adjency_matrix(self, adjency_matrix):
        """Set up the link dictionnary to represent the adjency matrix
        `adjency matrix` must be a self.size * self.size array of all trust values
        
        Warning: this does not overwrite existing links"""

        assert adjency_matrix.shape == (self.size, self.size)

        for i in range(self.size):
            for j in range(self.size):
                if adjency_matrix[i, j] > 0:
                    self.create_link(i, j)


"""Vertex class for handling people"""

class Vertex:

    valid_phenotypes = {"Trustful", "Random", "Optimist", "Pessimist", "Envious"}

    def __init__(self, phenotype, index, parameters):

        assert phenotype in self.valid_phenotypes

        self.size = parameters["Community size"]
        self.capacity = parameters["Cognitive capacity"]
        self.min_trust = parameters["Link minimum"]
        self.phenotype = phenotype
        self.index = index # Must not be changed
        self.load = 0
        self.trust = {}
        self.link = set()
    
    def create_link(self, end):
        """Add a link form the vertex to the vertex `end`"""
        self.link.add(end)
        
    def remove_link(self, end):
        """Remove the link form the vertex to the vertex `end` if it exists and do nothing otherwise"""
        if end in self.link:
            self.link.remove(end)
    
    def is_linked(self, end):
        """return True if there is a link with end"""
        return end in self.link

    def update_trust(self, other, increment):
        """Change the trust value of a link according to the response of the other and the expectations of each phenotypes"""
        
        new_load = self.load + increment
        if other in self.trust:
            self.trust[other] += increment
            if self.trust[other] <= 0:
                self.trust.pop(other)
        elif increment > 0:
            self.trust[other] = increment
        else:
            new_load = self.load

        if self.trust_in(other) >= self.min_trust:
            self.create_link(other)

        if new_load > self.capacity:
            diff = new_load - self.capacity
            while diff > 0:
                draw = np.random.randint(len(self.trust))
                drawn_vertex = list(self.trust.keys())[draw]
                
                
                if self.trust[drawn_vertex] >= diff:
                    self.trust[drawn_vertex] -= diff
                    if self.trust[drawn_vertex] <= 0:
                        self.trust.pop(drawn_vertex)
                    diff = 0
                else:
                    diff -= self.trust[drawn_vertex]
                    self.trust.pop(drawn_vertex)


                if self.is_linked(drawn_vertex) and self.trust_in(drawn_vertex) < self.min_trust:
                    self.remove_link(drawn_vertex)

        self.load = min(self.capacity, new_load)

    def trust_in(self, end):
        """Return trust value with the vertex `end`"""
        if end in self.trust:
            return self.trust[end]
        return 0

    def choose(self, other, game_matrix, temperature):
        """Return the choice of the person in the dyadic game based on temperature and phenotype
        0 -> cooperates
        1 -> defects
        Input:
        `other` vertex of the other interacting agent
        `game_matrix` the matrix of the game
        `temperature` the parameters of not following the predefined strategy
        """ 
        T = game_matrix[1, 0]
        S = game_matrix[0, 1]

        trust = self.trust_in(other)

        strategic_response = 1
        happy_response = None
        if self.phenotype == "Envious":
            if S - T >= 0:
                strategic_response = 0
            happy_response = 1 - strategic_response # Happy if gain more than the other
        elif self.phenotype == "Pessimist":
            if S > game_matrix[1, 1]:
                strategic_response = 0 
            if game_matrix[strategic_response, 0] > game_matrix[strategic_response, 1]:
                happy_response = 0
            else:
                happy_response = 1 # Happy if gain more than expected with his play
        elif self.phenotype == "Optimist":
            if T < game_matrix[0, 0]:
                strategic_response = 0
            sum0 = game_matrix[strategic_response, 0] + game_matrix[0, strategic_response] # Total payoff of both agent if the other cooperates
            sum1 = game_matrix[strategic_response, 1] + game_matrix[1, strategic_response] # Same but when the other defects
            if sum0 > sum1:
                happy_response = 0
            else:
                happy_response = 1 # Happy in case the sum of the game is maximum
        elif self.phenotype == "Trustful":
            strategic_response = 0
            happy_response = 0 # Happy if the other is indeed trustful
        elif self.phenotype == "Random":
            strategic_response = np.random.randint(2)
            happy_response = np.random.randint(2)
        
        if trust > self.min_trust:
            strategic_response = 0
            happy_response = 0

        if temperature == 0:
            return strategic_response, happy_response
        
        draw = np.random.rand()
        if draw > np.exp(- 1 / temperature):
            return strategic_response, happy_response

        return 1 - strategic_response, np.random.randint(2)
    
    def __str__(self):
        """Allows to print vertex in a readable way"""
        return self.phenotype + " " + str(self.index)

    def __hash__(self):
        return self.index