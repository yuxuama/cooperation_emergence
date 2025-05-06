"""graph.py
Implement the mathematical graph structure of the network
"""
import numpy as np
from operation import OperationStack
from utils import save_parameters, print_parameters, create_random_init, create_social_group
from tqdm import tqdm
  
"""Graph class for network structure"""

class Network:

    def __init__(self, parameters):
        # Define properties
        self.parameters = parameters
        self.max_iter = parameters["Number of interaction"]
        self.size = parameters["Community size"]
        self.min_trust = parameters["Link minimum"]
        self.temp = parameters["Temperature"]
        self.cognitive_capa = parameters["Cognitive capacity"]
        self.strategy_distrib = parameters["Strategy distributions"]
        self.out_dir = parameters["Output directory"]

        self.phenotypes = list(self.strategy_distrib.keys())
        self.distribution_grid = [0] # Used to initialized populations of each phenotypes according to parameters.
        for p in self.strategy_distrib.values():
            self.distribution_grid.append(self.distribution_grid[-1] + p)
        self.distribution_grid = self.size * np.array(self.distribution_grid)
         
        # Creating vertices
        self.vertices = np.zeros(self.size, dtype=Vertex)
        for i in range(self.size):
            self.create_vertex(i)

        # Operation
        self.oper = OperationStack()
        self.oper.activated = False

        # Initializing network structure
        self.initialize_structure()
    
    def create_vertex(self, index):
        """Add a person into the simulation"""
        # Choosing phenotypes according to distribution
        p = 0
        while index >= self.distribution_grid[p]:
            p += 1
        ph = self.phenotypes[p-1]
        if ph == "Random":
            self.vertices[index] = Random(index, self.parameters)
        if ph == "Trustful":
            self.vertices[index] = Trustful(index, self.parameters)
        if ph == "Pessimist":
            self.vertices[index] = Pessimist(index, self.parameters)
        if ph == "Optimist":
            self.vertices[index] = Optimist(index, self.parameters)
        if ph == "Envious":
            self.vertices[index] = Envious(index, self.parameters)

    def create_link(self, start, end):
        """Create a link between two vertices
        `start`: index or Vertex to start the link with
        `end`: index or Vertex to end the link with"""
        if type(start) == int:
            start = self.vertices[start]
        if type(end) == int:
            end = self.vertices[end]
        start.create_link(end)
        self.oper.add_link(start.index, end.index)     
    
    def remove_link(self, start, end):
        """Remove a link between two vertices
        `start`: index or Vertex to start the link with
        `end`: index or Vertex to end the link with"""
        if type(start) == int:
            start = self.vertices[start]
        if type(end) == int:
            end = self.vertices[end]
        start.remove_link(end)
        self.oper.remove_link(start.index, end.index)

    def is_linked(self, start, end):
        """return True if start and end are linked"""
        if type(start) == int:
            start = self.vertices[start]
        if type(end) == int:
            end = self.vertices[end]
        start.is_linked(end)

    def get_trust(self, start, end):
        """return the trust value between start and end
        `start`: index or Vertex to start the link with
        `end`: index or Vertex to end the link with"""
        if type(start) == int:
            start = self.vertices[start]
        if type(end) == int:
            end = self.vertices[end]
            
        return start.trust_in(end)
    
    def set_trust(self, start, end, trust):
        """Set the trust value of the trust edge starting from `start` and endind at `end` to `trust`"""
        if type(start) == int:
            start = self.vertices[start]
        if type(end) == int:
            end = self.vertices[end]
        start.set_trust(end, trust)
    
    def reset(self, new_parameters):
        self.__init__(new_parameters)
    
    def initialize_structure(self):
        """initialize the structure of the network according to the parameters"""
        mode = self.parameters["Init"]
        if mode == "Random":
            self.init_random()
        elif mode == "Groups":
            if "Number of groups" in self.parameters:
                self.init_groups(self.parameters["Number of groups"])
            else:
                print("INFO: parameter 'Number of groups' found in parameter file")
                default_value = self.size // (self.cognitive_capa + 1)
                print("INFO: Using default value: {}".format(default_value))
                self.init_groups(default_value)
        elif mode != "Empty":
            raise KeyError("The initial condition defined by the 'Init' parameters is unknown")

    def init_random(self):
        """Initialize the network with random trust values"""
        trust_adjacency_matrix = np.zeros((self.size, self.size))
        create_random_init(trust_adjacency_matrix, self.cognitive_capa, self.min_trust)
        self.set_adjacency_trust_matrix(trust_adjacency_matrix)

    def init_groups(self, n_group):
        """Initialize the network with `n_group` groups
        If the size of the network is not divisible by `n_group` then
        creates a smaller group at the end"""
        group_size = self.size // n_group
        if group_size - 1 > self.cognitive_capa:
            raise ValueError("Impossible to create {0} fully connected group with cognitive capacity {1} and network of size {2}".format(n_group, self.cognitive_capa, self.size))
        group_rest = self.size % n_group
        sizes = np.ones(n_group, dtype=int) * group_size
        if group_rest != 0:
            print("WARNING: size is not divisible by {}".format(n_group))
            print("  |_ {} agent will not be in a group".format(group_rest))
        assignation = np.zeros(self.size, dtype=bool)
        trust_adjacency_matrix = np.zeros((self.size, self.size))
        for i in range(n_group):
            create_social_group(sizes[i], assignation, trust_adjacency_matrix, self.min_trust)
        self.set_adjacency_trust_matrix(trust_adjacency_matrix)

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

        if happy1 is None:
            happy1 = choice2
        if happy2 is None:
            happy2 = choice1
        
        if choice2 == happy1:
            v1.update_trust(v2, gain, self.oper)
        else:
            v1.update_trust(v2, loss, self.oper)
        
        if choice1 == happy2:
            v2.update_trust(v1, gain, self.oper)
        else:
            v2.update_trust(v1, loss, self.oper)

    def play(self):
        """Run the simulation
        Return the OperationStack object which contains all the history of the simulation"""
        print_parameters(self.parameters)
        
        self.oper.activated = True # Activate write mode of the OperationStack
        self.oper.set_link_from_array(self.get_adjacency_link_matrix())
        self.oper.set_trust_from_array(self.get_adjacency_trust_matrix())
        for _ in tqdm(range(self.max_iter)):
            self.interact()
            self.oper.next_iter()
        # Save
        self.oper.activated = False # Deactivate write mode
        self.oper.save(self.out_dir)
        save_parameters(self.parameters, self.out_dir)
        return self.oper

    def get_adjacency_trust_matrix(self):
        """Return the adjacency matrix of all trust"""
        trust_adjacency_matrix = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                trust_adjacency_matrix[i, j] = self.get_trust(i, j)
        return trust_adjacency_matrix

    def set_adjacency_trust_matrix(self, adjacency_matrix):
        """Set all trust value to correspond to the adjacency matrix
        The links are updated accordingly automatically"""
        assert adjacency_matrix.shape == (self.size, self.size)
        assert np.max(np.sum(adjacency_matrix, axis=1)) <= self.cognitive_capa
        for i in range(self.size):
            for j in range(self.size):
                if adjacency_matrix[i, j] > 0:
                    self.set_trust(i, j, adjacency_matrix[i, j])

    def get_adjacency_link_matrix(self):
        """Return the adjacency matrix of all link"""
        link_adjacency_matrix = np.zeros((self.size, self.size), dtype=bool)
        for i in range(self.size):
            v = self.vertices[i]
            for vend in v.link:
                link_adjacency_matrix[i, vend.index] = True

        return link_adjacency_matrix       

    def generate_output_name(self):
        """Generate a name that represent uniquely this network
        format: '<Heurisitc>_<Distribution>_I<Init>_L<Link minimum>_C<Cognitive capacity>_S<size>_T<temperature>
        distribution format: "Envious: 0.3" -> E3"""
        name = ""
        name += self.parameters["Heuristic"]
        name += "_"
        # distribution
        distribution = ""
        keys = list(self.parameters["Strategy distributions"].keys())
        keys = sorted(keys)
        for key in keys:
            distribution += key[0] + str(self.parameters["Strategy distributions"][key])[2::]
        name += distribution
        
        # other
        name += "_"
        if "Number of groups" in self.parameters and self.parameters["Init"] == "Groups":
            name += str(self.parameters["Number of groups"])
        name += self.parameters["Init"]
        name += "_"
        name += "L" + str(self.min_trust) + "_"
        name += "C" + str(self.cognitive_capa) + "_"
        name += "S" + str(self.size) + "_"
        name += "T" + str(self.temp)
        return name




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

    def update_trust(self, other, increment, oper):
        """Change the trust value of a link according to the response of the other and the expectations of each phenotypes"""
        
        new_load = self.load + increment
        if other in self.trust:
            oper.increment_trust(self.index, other.index, max(-self.trust[other], increment))
            self.trust[other] += increment
            if self.trust[other] <= 0:
                self.trust.pop(other)
        elif increment > 0:
            self.trust[other] = increment
            oper.increment_trust(self.index, other.index, increment)
        else:
            new_load = self.load

        if other not in self.link and self.trust_in(other) >= self.min_trust:
            self.create_link(other)
            oper.add_link(self.index, other.index)

        if new_load > self.capacity:
            diff = new_load - self.capacity
            while diff > 0:
                draw = np.random.randint(len(self.trust))
                drawn_vertex = list(self.trust.keys())[draw]
                
                if self.trust[drawn_vertex] > diff:
                    self.trust[drawn_vertex] -= diff
                    oper.increment_trust(self.index, drawn_vertex.index, -diff)
                    diff = 0
                else:
                    diff -= self.trust[drawn_vertex]
                    oper.increment_trust(self.index, drawn_vertex.index, -self.trust[drawn_vertex])
                    self.trust.pop(drawn_vertex)

                if self.is_linked(drawn_vertex) and self.trust_in(drawn_vertex) < self.min_trust:
                    self.remove_link(drawn_vertex)
                    oper.remove_link(self.index, drawn_vertex.index)

        self.load = min(self.capacity, new_load)

    def trust_in(self, end):
        """Return trust value with the vertex `end`"""
        if end in self.trust:
            return self.trust[end]
        return 0

    def set_trust(self, end, trust):
        self.trust[end] = trust
        if trust >= self.min_trust:
            self.create_link(end)
        self.load += trust
            
    def __str__(self):
        """Allows to print vertex in a readable way"""
        return self.phenotype + " " + str(self.index)

    def __hash__(self):
        return self.index

class Random(Vertex):

    def __init__(self, index, parameters):
        super().__init__("Random", index, parameters)
        self.heuristic = parameters["Heuristic"]
    
    def choose(self, other, game_matrix, temperature):
        """Return the choice of the agent according to the game"""
        strategic_response = 1
        happy_response = None
        if self.heuristic == "RTH":
            strategic_response, happy_response = np.random.randint(2), 0
        elif self.heuristic == "SITH":
            strategic_response, happy_response = np.random.randint(2), np.random.randint(2)
        elif self.heuristic == "Complex":
            strategic_response, happy_response = np.random.randint(2), np.random.randint(2)

        if self.trust_in(other) > self.capacity:
            strategic_response = 0
            happy_response = 0
        
        if temperature == 0:
            return strategic_response, happy_response
        draw = np.random.rand()
        if draw > np.exp(- 1 / temperature):
            return strategic_response, happy_response

        return 1 - strategic_response, np.random.randint(2)

class Trustful(Vertex):

    def __init__(self, index, parameters):
        super().__init__("Trustful", index, parameters)
        self.heuristic = parameters["Heuristic"]
    
    def choose(self, other, game_matrix, temperature):
        """Return the choice of the agent according to the game"""
        strategic_response = 0
        happy_response = 0
        # Temperature noise
        if temperature == 0:
            return strategic_response, happy_response
        draw = np.random.rand()
        if draw > np.exp(- 1 / temperature):
            return strategic_response, happy_response

        return 1 - strategic_response, np.random.randint(2)

class Pessimist(Vertex):

    def __init__(self, index, parameters):
        super().__init__("Pessimist", index, parameters)
        self.heuristic = parameters["Heuristic"]
    
    def choose(self, other, game_matrix, temperature):
        """Return the choice of the agent according to the game"""
        strategic_response = 1
        happy_response = None
        if self.trust_in(other) > self.min_trust or game_matrix[0, 1] > game_matrix[1, 1]:
            strategic_response = 0
        if self.trust_in(other) > self.min_trust:
            happy_response = 0
        elif self.heuristic == "RTH":
            happy_response = 0
        elif self.heuristic == "SITH":
            happy_response = None
        elif self.heuristic == "Complex":
            if game_matrix[strategic_response, 0] > game_matrix[strategic_response, 1]:
                happy_response = 0
            else:
                happy_response = 1 # Happy if gain more than expected with his play

        # Temperature noise
        if temperature == 0:
            return strategic_response, happy_response
        draw = np.random.rand()
        if draw > np.exp(- 1 / temperature):
            return strategic_response, happy_response

        return 1 - strategic_response, np.random.randint(2)

class Optimist(Vertex):

    def __init__(self, index, parameters):
        super().__init__("Optimist", index, parameters)
        self.heuristic = parameters["Heuristic"]
    
    def choose(self, other, game_matrix, temperature):
        """Return the choice of the agent according to the game"""
        strategic_response = 1
        happy_response = None
        if self.trust_in(other) > self.min_trust or game_matrix[0, 1] < game_matrix[0, 0]:
            strategic_response = 0
        if self.trust_in(other) > self.min_trust:
            happy_response = 0
        elif self.heuristic == "RTH":
            happy_response = 0
        elif self.heuristic == "SITH":
            happy_response = 0
        elif self.heuristic == "Complex":
            sum0 = game_matrix[strategic_response, 0] + game_matrix[0, strategic_response] # Total payoff of both agent if the other cooperates
            sum1 = game_matrix[strategic_response, 1] + game_matrix[1, strategic_response] # Same but when the other defects
            if sum0 > sum1:
                happy_response = 0
            else:
                happy_response = 1 # Happy in case the sum of the game is maximum

        # Temperature noise
        if temperature == 0:
            return strategic_response, happy_response
        draw = np.random.rand()
        if draw > np.exp(- 1 / temperature):
            return strategic_response, happy_response

        return 1 - strategic_response, np.random.randint(2)

class Envious(Vertex):

    def __init__(self, index, parameters):
        super().__init__("Envious", index, parameters)
        self.heuristic = parameters["Heuristic"]
    
    def choose(self, other, game_matrix, temperature):
        """Return the choice of the agent according to the game"""
        strategic_response = 1
        happy_response = None
        if self.trust_in(other) > self.min_trust or game_matrix[0, 1] >= game_matrix[1, 0]:
            strategic_response = 0
        if self.trust_in(other) > self.min_trust:
            happy_response = 0
        elif self.heuristic == "RTH":
            happy_response = 0
        elif self.heuristic == "SITH":
            happy_response = 1 - strategic_response
        elif self.heuristic == "Complex":
            happy_response = 1 - strategic_response

        # Temperature noise
        if temperature == 0:
            return strategic_response, happy_response
        draw = np.random.rand()
        if draw > np.exp(- 1 / temperature):
            return strategic_response, happy_response

        return 1 - strategic_response, np.random.randint(2)