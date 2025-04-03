# Model Framework for Trust-Driven Cooperation and Social Capital Distribution

This model intends to reproduce the social structure observed in Migliano et al. (2017) and Musciotto et al. (2023). The starting point is the experimental paper Poncela-Casasnovas et al. (2016), where subjects had to interact randomly with other people through different games, and four key heuristics governing their behavior were found. Those experiments were carried out in a well-mixed population context, meaning that every subject could interact with every other one with equal probability. This model brings the network
formation problem into that context. The other ingredient of the model is the limited capability to keep personal relationships, akin to the notion of Dunbar’s number as discussed in Tamarit et al. (2018). There is a maximum number of relationships one can have that depends on how much effort/trust/time/cognitive capability is devoted to each of them.

# How to use

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