"""dataset.py
Implement the dataset structure
"""
from graph import Network
from networkx import from_numpy_array, betweenness_centrality
from analysis import compute_xhi
import numpy as np

def selector_parser(selector):
    """Parse selector
    Selector format specification: must be key or list of key"""
    pass

class Dataset:

    def __init__(self, name="Dataset"):
        self.name = str(name)
        self.size = 0
        self.data = {}
    
    def init_with_network(self, net: Network):
        
        self.size = net.parameters["Community size"]
        vertices = net.vertices
        link_adjacency = net.get_link_adjacency_matrix()
        kxgraph_link = from_numpy_array(link_adjacency)
        centralities = betweenness_centrality(kxgraph_link)
        out_degrees = np.sum(link_adjacency, axis=1)
        in_degrees = np.sum(link_adjacency, axis=0)
        for i in range(self.size):
            i_data = {}
            # Phenotype
            i_data["Phenotype"] = vertices[i].phenotype
            # Load
            i_data["Load"] = vertices[i].load / net.parameters["Cognitive capacity"]
            # Outdegree
            i_data["Out degree"] = out_degrees[i]
            # Indegree
            i_data["In degree"] = in_degrees[i]
            # Centrality
            i_data["Centrality"] = centralities[i]
            # Asymmetry

            # Eta        

            # Adding field to dataset data
            self.data[vertices[i].index] = i_data

    def init_with_stack(self, stackdir):
        pass

    def init_with_hdf5(self, filpath):
        pass

    def load_from_file(self, filename):
        pass

    def add_id(self, id, data_dict={}):
        self.size += 1
        if id in self.data:
            print("WARNING: Erasing previous data")
        self.data[id] = data_dict
    
    def add_field_in_id(self, id, key, value):
        if not id in self.data:
            self.add_id(id, {key: value})
            return
        self.data[id][key] = value

    def get(self, id, fields="All"):
        query = Dataset(id)
        if type(fields) is str:
            if fields == "All":
                query.add_id(id, self.data[id])
                return query
            if fields in self.data[id]:
                query.add_field_in_id(id, fields, self.data[id][fields])
                return query
            raise KeyError("Unknown field must be in {0}".format(self.data[id].keys()))
        
        try:
            iterator = iter(fields)
        except TypeError:
            raise TypeError("`fields` attribute is not iterable and not a `str`")
        
        for field in iterator:
            if field in self.data[id]:
                query.add_field_in_id(id, field, self.data[id][field])
            else:
                raise KeyError("Unknown field must be in {0}".format(self.data[id].keys()))
        return query

    def get_all(self, fields="All"):
        pass

    def aggregate(self, fields):
        query = Dataset("agregated " + self.name + str(fields))
        for i in range(self.size):
            data = self.get(i, fields).data
            for key in data.keys():
                if key in query.data:
                    query.data[key][i] = data[key]
                else:
                    query.data[key] = {i: data[key]}
        return query

    def group_by(self, classifier):
        pass

    def copy(self):
        copyds = Dataset(self.name)
        copyds.data = self.data.copy()
        copyds.size = self.size
        return copyds

    def save(self, filename):
        pass

    def __str__(self):
        return "Dataset '" + self.name + "' with data: \n" + str(self.data)


class DatasetGroup:

    def __init__(self, name):
        self.name = name
        self.size = 0
        self.containt = {}
    
    def add_dataset(self, dataset):
        self.containt[dataset.name] = dataset
    
    def get(self, id, selector):
        pass

    def get_all(self, selector):
        pass

