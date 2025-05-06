# Model Framework for Trust-Driven Cooperation and Social Capital Distribution

This model intends to reproduce the social structure observed in Migliano et al. (2017) and Musciotto et al. (2023). The starting point is the experimental paper Poncela-Casasnovas et al. (2016), where subjects had to interact randomly with other people through different games, and four key heuristics governing their behavior were found. Those experiments were carried out in a well-mixed population context, meaning that every subject could interact with every other one with equal probability. This model brings the network
formation problem into that context. The other ingredient of the model is the limited capability to keep personal relationships, akin to the notion of Dunbar’s number as discussed in Tamarit et al. (2018). There is a maximum number of relationships one can have that depends on how much effort/trust/time/cognitive capability is devoted to each of them.

# Simulation parameters description
**Simulation parameters**
- Number of interaction: number of interaction simulated
- Output directory: root directory for saving the network
- Save mode: 3 differents save modes
  - `"Stack"`: save with `OperationStack` structure implemented in `operation.py`
  - `"Last"`: save only the final state in a `.h5` file
  - `"Off"`: don't save
- Verbose (optional: default=True): if True display information when the network is runned

**Initial state**
- Init: 3 different initializations
  -`"Empty"`: no trust values
  -`"Random"`: random trust values
  -`"Groups"`: create fully connected disjoint groups
- Number of groups (optional): Number of groups when Init is `"Groups"`. Default take the biggest size of group possible according to other parameters such as Size and Cognitive capacity

**Model parameters**
- Community size: number of agent in the simulation
- Link minimum: minimum trust value for a social tie to exist
- Cognitive capacity: maximum out degree of a node (of a agent)
- Heuristic:
  - `"RTH"`: Real Trust Heuristic
  - `"SITH"`: Self Interest Trust Heuristic
- Temperature: parameter controling to what extent an agent follow its predefined strategy
- Strategy distribution:
  - Envious (optional): proportion of Envious phenotype
  - Optimist (optional): proportion of Optimist phenotype
  - Pessimist (optional): proportion of Pessimist phenotype
  - Random (optional): proportion of Random phenotype
  - Trustful (optional): proportion of Trustful phenotype
  
# How to run a simulation

- Enter parameters of the simulation in the `parameters.yaml` file.
- All is then taken care of by the `Network` object defined in the `graph.py` file

Launching a simulation:
```py
from graph import Network
from utils import parse_parameters

yaml_file = r"<path to yaml file>"

parameters = parse_parameters(yaml_file) # Load parameters from the file
net = Network(parameters) # Define the Network object
net.play() # Run the simulation
```

# How to analyze a simulation saved with mode "Stacks"

In the directory where the simulation is saved you will find 3 differents files:
- `parameters.yaml`: a copy of the parameters used for the simulation
- `init.h5`: a HDF5 file storing the initial adjacency matrix for both trust and link
  - trust network in dataset `"Trust"`
  - link network in dataset `"Link"`
- `stack.csv`: historic of all updates of the network for each interaction
  - Format: |Interaction Number|Operation name|i|j|value|

State of the network for different number of interaction

```py
from operation import OperationStack

out_dir = r"<path to out directory>" # Where the simulation is saved

oper = OperationStack(out_dir)
trust_adjacency_matrix, link_adjacency_matrix = oper.resolve(i) # Return the state of the network at the i-th interaction
final_trust_adjacency_matrix, final_link_adjacency_matrix = oper.resolve(-1) # Equivalent to `oper.resolve(oper.iter_number)`
```
