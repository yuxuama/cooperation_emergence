"""graph.py
Implement the mathematical graph structure of the network
"""
import numpy as np
from operation import OperationStack
from utils import save_parameters, print_parameters, \
    create_random_init, create_social_group, \
    parse_parameters, generate_output_name_from_parameters, \
    proceed, save_parameters_in_hdf5, extract_parameters_from_hdf5
from tqdm import tqdm
import h5py
import os

"""Correspondence table"""
p_niter = "Number of interaction"
p_size = "Community size"
p_link_min = "Link minimum"
p_cognitive_c = "Cognitive capacity"
p_T = "Temperature"

  
"""Graph class for network structure"""

class Network:

    def __init__(self, seed=None):
        """`parameters:` parameters dict from a YAML parameters file
        `seed`: the network is identified merely with the paramters. A seed (must be stringfiable) allows discrimination"""
        # Define properties
        self.parameters = None
        self.verbose = True
        self.seed = seed
        self.out_dir = None
        self.phenotypes_table = None # Correspondance index-phenotype
        self.community_table = None # Correspondence index-community (for future work)
        self.vertices = None
        self.oper = OperationStack()
        self.name = ""

        self._reload_context = False
        self._first_stack_context = True
    
    def init_with_parameters(self, parameters):
        """Initialize the Graph structure with only parameters"""
        # Initialize properties
        self.parameters = parameters
        self.out_dir = parameters["Output directory"]
        if "Verbose" in self.parameters:
            self.verbose = parameters["Verbose"]
        self.name = self.generate_name(self.seed)

        # Create vertices with parameters
        self.create_all_vertices()
        
        # Operation
        self.oper = OperationStack()
        self.oper.activated = False

        # Initializing network structure using parameters
        self.initialize_structure()
        self.name = self.generate_name(self.seed)

    def reload_with_hdf5(self, filepath):
        """Initialize the Graph structure with the HDF5 file at `filepath`"""
        # Load file
        self.parameters = extract_parameters_from_hdf5(filepath)
        
        # Initialize properties
        self._reload_context = True
        self.out_dir = self.parameters["Output directory"]
        if "Verbose" in self.parameters:
            self.verbose = self.parameters["Verbose"]
        self.name = self.generate_name(self.seed)

        # Create vertices
        self.create_all_vertices()

        # Initializing structure
        h5file = h5py.File(filepath)
        t = h5file["Trust"]
        self.set_trust_adjacency_matrix(t)

    def reload_with_stack(self, stackdir, inter=-1):
        """Initialize the Graph object with the OperationStack saved in `stackdir`"""
        # Load stack
        self.oper = OperationStack(stackdir)
        self.oper = self.oper.copy_until(inter)
        self.oper.activated = False

        # Initialize properties
        self._reload_context = True
        self.parameters = parse_parameters(stackdir + "parameters.yaml")
        self.out_dir = self.parameters["Output directory"]
        if "Verbose" in self.parameters:
            self.verbose = self.parameters["Verbose"]
        self.name = self.generate_name(self.seed)
        
        # Create vertices
        self.create_all_vertices()

        # Initializing structure
        t, _ = self.oper.resolve(-1)
        self.parameters[p_niter] = self.oper.read_interaction - 1
        self.set_trust_adjacency_matrix(t)
    
    def set_save_mode(self, new_save_mode):
        """Change saving mode"""
        if new_save_mode == self.parameters["Save mode"]:
            print("Same saving parameters detected")
            return
        old = self.parameters["Save mode"]
        self.parameters["Save mode"] = new_save_mode
        if new_save_mode == "Stack":
            self._first_stack_context = True
        print("Successfully change saving mode: {0} --> {1}".format(old, new_save_mode))
    
    def create_all_vertices(self):
        """Create all vertices of the graph according to the parameters"""
        # Phenotypes distribution
        strategy_distrib = self.parameters["Strategy distributions"]
        possible_phenotypes = list(strategy_distrib.keys())
        distribution_grid = [0] # Used to initialized populations of each phenotypes according to parameters.
        for p in strategy_distrib.values():
            distribution_grid.append(distribution_grid[-1] + p)
        distribution_grid = self.parameters[p_size] * np.array(distribution_grid)
         
        # Creating vertices
        self.vertices = np.zeros(self.parameters[p_size], dtype=Vertex)
        self.phenotypes_table = ["" for i in range(self.parameters[p_size])]
        d_pointer = 1
        ph_pointer = 0
        for i in range(self.parameters[p_size]):
            if i+1 > distribution_grid[d_pointer]:
                d_pointer += 1
                ph_pointer += 1
            phenotype = possible_phenotypes[ph_pointer]
            self.phenotypes_table[i] = phenotype
            self.create_vertex(i, phenotype)

    def create_vertex(self, index, phenotype):
        """Add a person into the simulation"""
        # Choosing phenotypes according to distribution
        if phenotype == "Random":
            self.vertices[index] = Random(index, self.parameters)
        elif phenotype == "Trustful":
            self.vertices[index] = Trustful(index, self.parameters)
        elif phenotype == "Pessimist":
            self.vertices[index] = Pessimist(index, self.parameters)
        elif phenotype == "Optimist":
            self.vertices[index] = Optimist(index, self.parameters)
        elif phenotype == "Envious":
            self.vertices[index] = Envious(index, self.parameters)
        else:
            raise KeyError("Unsupported phenotype {} in parameters".format(phenotype))

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
    
    def reset(self, new_seed=None):
        self.__init__(new_seed)
    
    def generate_name(self, seed):
        """Generate name of the simulation based on the parameters"""
        postfix = ""
        if "Seed" in self.parameters:
            seed = self.parameters["Seed"]
        elif self.seed is not None:
            seed = self.seed
            self.parameters["Seed"] = seed
        if self.parameters["Save mode"] == "Last":
            postfix += str(self.parameters[p_niter])
            if seed is not None:
                postfix += "_" + str(seed)
        return generate_output_name_from_parameters(self.parameters, postfix)

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
                default_value = self.parameters[p_size] // (self.parameters[p_cognitive_c] + 1)
                print("INFO: Using default value: {}".format(default_value))
                self.init_groups(default_value)
        elif mode != "Empty":
            raise KeyError("The initial condition defined by the 'Init' parameters is unknown")

    def init_random(self):
        """Initialize the network with random trust values"""
        trust_adjacency_matrix = np.zeros((self.parameters[p_size], self.parameters[p_size]))
        create_random_init(trust_adjacency_matrix, self.parameters[p_cognitive_c], self.parameters[p_link_min])
        self.set_trust_adjacency_matrix(trust_adjacency_matrix)

    def init_groups(self, n_group):
        """Initialize the network with `n_group` groups
        If the size of the network is not divisible by `n_group` then
        creates a smaller group at the end"""
        group_size = self.parameters[p_size] // n_group
        if group_size - 1 > self.parameters[p_cognitive_c]:
            raise ValueError("Impossible to create {0} fully connected group with cognitive capacity {1} and network of size {2}".format(n_group, self.parameters[p_cognitive_c], self.parameters[p_size]))
        group_rest = self.parameters[p_size] % n_group
        sizes = np.ones(n_group, dtype=int) * group_size
        if group_rest != 0:
            print("WARNING: size is not divisible by {}".format(n_group))
            print("  |_ {} agent will not be in a group".format(group_rest))
        assignation = np.zeros(self.parameters[p_size], dtype=bool)
        trust_adjacency_matrix = np.zeros((self.parameters[p_size], self.parameters[p_size]))
        
        for i in range(n_group):
            create_social_group(sizes[i], assignation, trust_adjacency_matrix, self.parameters[p_link_min])
        self.set_trust_adjacency_matrix(trust_adjacency_matrix)

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
        
        pair = np.random.choice(self.parameters[p_size], 2, replace=False)
        T = np.random.rand()*10 + 5 # Random number between 5 and 15
        S = np.random.rand()*10 # Random number between 0 and 10

        game_matrix = np.array([[10, S], [T, 5]])
        
        v1 = self.vertices[pair[0]]
        v2 = self.vertices[pair[1]]
        
        choice1, happy1 = v1.choose(v2, game_matrix, self.parameters[p_T])
        choice2, happy2 = v2.choose(v1, game_matrix, self.parameters[p_T])

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

    def play(self, niter=0):
        """Run the simulation
        Return the OperationStack object which contains all the history of the simulation"""

        if self.parameters is None:
            raise AttributeError("Graph has no parameters: cannot launch a simulation")

        if not self._reload_context:
            niter = self.parameters[p_niter]
        elif niter == 0:
            answer = proceed("You have not specified a number of interaction to simulate after the reload." \
                             "Value of {} will be taken from the parameters".format(self.parameters[p_niter]))
            if not answer:
                return
            niter = self.parameters[p_niter]
            self.parameters[p_niter] *= 2
        else:
            self.parameters[p_niter] += niter
        self.name = self.generate_name(self.seed)

        def idle(arg):
            return arg
        selector = idle
        if self.verbose:
            print_parameters(self.parameters)
            selector = tqdm
        if self.parameters["Save mode"] == "Stack":
            self.oper.activated = True # Activate write mode of the OperationStack
            if self._first_stack_context:
                self.oper.set_link_from_array(self.get_link_adjacency_matrix())
                self.oper.set_trust_from_array(self.get_trust_adjacency_matrix())

        # Main loop
        for _ in selector(range(niter)):
            self.interact()
            self.oper.next_iter()
        self.oper.activated = False # Deactivate write mode

        # Saving
        self.save()
        self._reload_context = True
        if self._first_stack_context:
            self._first_stack_context = False

    def save(self):
        """Handle save of the network"""
        if self.parameters["Save mode"] == "Stack":
            if self.verbose:
                print("Saving in: ", self.out_dir + self.name)
            out = self.out_dir + self.name + "/"
            self.oper.save(out, reload_context=self._reload_context)
            save_parameters(self.parameters, out)
        elif self.parameters["Save mode"] == "Last":
            os.makedirs(self.out_dir, exist_ok=True)
            h5file = h5py.File(self.out_dir + self.name + ".h5", "w")
            h5file["Trust"] = self.get_trust_adjacency_matrix()
            h5file["Link"] = self.get_link_adjacency_matrix()
            save_parameters_in_hdf5(self.parameters, h5file)
        elif self.parameters["Save mode"] != "Off":
            raise ValueError("The 'Save mode' parameter used is not handled")

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

    def get_trust_adjacency_matrix(self):
        """Return the adjacency matrix of all trust"""
        trust_adjacency_matrix = np.zeros((self.parameters[p_size], self.parameters[p_size]))
        for i in range(self.parameters[p_size]):
            for j in range(self.parameters[p_size]):
                trust_adjacency_matrix[i, j] = self.get_trust(i, j)
        return trust_adjacency_matrix

    def set_trust_adjacency_matrix(self, adjacency_matrix):
        """Set all trust value to correspond to the adjacency matrix
        The links are updated accordingly automatically"""
        assert adjacency_matrix.shape == (self.parameters[p_size], self.parameters[p_size])
        assert np.max(np.sum(adjacency_matrix, axis=1)) <= self.parameters[p_cognitive_c]
        for i in range(self.parameters[p_size]):
            for j in range(self.parameters[p_size]):
                if adjacency_matrix[i, j] > 0:
                    self.set_trust(i, j, adjacency_matrix[i, j])

    def get_link_adjacency_matrix(self):
        """Return the adjacency matrix of all link"""
        link_adjacency_matrix = np.zeros((self.parameters[p_size], self.parameters[p_size]), dtype=bool)
        for i in range(self.parameters[p_size]):
            v = self.vertices[i]
            for vend in v.link:
                link_adjacency_matrix[i, vend.index] = True

        return link_adjacency_matrix       

"""Vertex class for handling people"""

class Vertex:

    valid_phenotypes = {"Trustful", "Random", "Optimist", "Pessimist", "Envious"}

    def __init__(self, phenotype, index, parameters):

        assert phenotype in self.valid_phenotypes

        self.size = parameters["Community size"]
        self.capacity = parameters["Cognitive capacity"]
        self.link_min = parameters["Link minimum"]
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

        if other not in self.link and self.trust_in(other) >= self.link_min:
            self.create_link(other)
            oper.add_link(self.index, other.index)
        if other in self.link and self.trust_in(other) < self.link_min:
            self.remove_link(other)
            oper.remove_link(self.index, other.index)

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

                if self.is_linked(drawn_vertex) and self.trust_in(drawn_vertex) < self.link_min:
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
        if trust >= self.link_min:
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
        if self.trust_in(other) > self.link_min or game_matrix[0, 1] > game_matrix[1, 1]:
            strategic_response = 0
        if self.trust_in(other) > self.link_min:
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
        if self.trust_in(other) > self.link_min or game_matrix[0, 1] < game_matrix[0, 0]:
            strategic_response = 0
        if self.trust_in(other) > self.link_min:
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
        if self.trust_in(other) > self.link_min or game_matrix[0, 1] >= game_matrix[1, 0]:
            strategic_response = 0
        if self.trust_in(other) > self.link_min:
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