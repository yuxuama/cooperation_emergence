"""dataset.py
Implement the dataset structure
"""
from graph import Network
from networkx import from_numpy_array, betweenness_centrality
from analysis import measure_etas_from_xhi, measure_individual_asymmetry
from utils import get_vertex_distribution
import numpy as np


################################################################################################
# Dataset utils
################################################################################################


def selector_parser(selector):
    """Parse selector for more selection freedom (useful ?)"""
    pass


################################################################################################
# Dataset class
################################################################################################

class Dataset:

    def __init__(self, name="Dataset", niter=0):
        self.name = name
        self.size = 0
        self.data = {}
        self.niter = niter
    
    def init_with_network(self, net: Network):
        """Init Dataset with all local values of the network"""
        self.size = net.parameters["Community size"]
        self.niter = net.parameters["Number of interaction"]
        vertices = net.vertices
        link_adjacency = net.get_link_adjacency_matrix()
        trust_adjacency = net.get_trust_adjacency_matrix()
        kxgraph_link = from_numpy_array(link_adjacency)
        centralities = betweenness_centrality(kxgraph_link)
        out_degrees = np.sum(link_adjacency, axis=1)
        in_degrees = np.sum(link_adjacency, axis=0)
        asymmetries = measure_individual_asymmetry(link_adjacency, full=True)
        for i in range(self.size):
            i_data = {}
            # Graph index
            index = vertices[i].index
            # Phenotype
            i_data["Phenotype"] = vertices[i].phenotype
            # Load
            i_data["Load"] = vertices[i].load / net.parameters["Cognitive capacity"]
            # Outdegree
            i_data["Out degree"] = out_degrees[index]
            # Indegree
            i_data["In degree"] = in_degrees[index]
            # Centrality
            i_data["Centrality"] = centralities[index]
            # Asymmetry
            i_data["Asymmetry"] = asymmetries[index]
            # Eta        
            i_data["Eta"] = measure_etas_from_xhi(index, trust_adjacency, net.parameters)
            # Trust histogram

            # bins

            # Adding field to dataset data
            self.data[vertices[i].index] = i_data
        return self

    def init_with_stack(self, stackdir):
        """Initialize Dataset with a Stack"""
        net = Network()
        net.reload_with_stack(stackdir)
        return self.init_with_network(net)

    def init_with_hdf5(self, filepath):
        """Initialize Dataset with a HDF5 file"""
        net = Network()
        net.reload_with_hdf5(filepath)
        return self.init_with_network(net)

    def init_with_matrices(self, link_adjacency, trust_adjacency, parameters, niter):
        """Initialize structure for local measurement only with matrices and parameters"""
        self.size = parameters["Community size"]
        self.niter = niter
        kxgraph_link = from_numpy_array(link_adjacency)
        centralities = betweenness_centrality(kxgraph_link)
        out_degrees = np.sum(link_adjacency, axis=1)
        in_degrees = np.sum(link_adjacency, axis=0)
        asymmetries = measure_individual_asymmetry(link_adjacency, full=True)
        phenotype_table = get_vertex_distribution(parameters)
        for i in range(self.size):
            i_data = {}
            # Phenotype
            i_data["Phenotype"] = phenotype_table[i]
            # Load
            i_data["Load"] = np.sum(trust_adjacency[i]) / parameters["Cognitive capacity"]
            # Outdegree
            i_data["Out degree"] = out_degrees[i]
            # Indegree
            i_data["In degree"] = in_degrees[i]
            # Centrality
            i_data["Centrality"] = centralities[i]
            # Asymmetry
            i_data["Asymmetry"] = asymmetries[i]
            # Eta        
            i_data["Eta"] = measure_etas_from_xhi(i, trust_adjacency, parameters)
            # Adding field to dataset data
            self.data[i] = i_data

    def add(self, id, data_dict={}):
        """add field in dataset"""
        if id in self.data:
            print("WARNING: you are replacing already existing `id` be cautious")
        self.data[id] = data_dict
        self.size += 1
    
    def add_field_in_id(self, id, key, value):
        """Add field in the dictionnary stored in id `id`"""
        if not id in self.data:
            self.add_id(id, {key: value})
            return
        self.data[id][key] = value
    
    def modify_field_in_id(self, id, key, new_value, mode="="):
        """Modify the field `field` for id `id` with `new_value
        Possible modes: `"="`, `"+"`, `"-"`, `"*"`, `"/"`"""
        if mode == "=":
            self.data[id][key] = new_value
        elif mode == "+":
            self.data[id][key] += new_value
        elif mode == "-":
            self.data[id][key] -= new_value
        elif mode == "*":
            self.data[id][key] *= new_value
        elif mode == "/":
            self.data[id][key] /= new_value

    def get_item(self, id, query="All"):
        """Get item of id `id` in the dataset matching `query` requirment"""
        output = {}
        if type(query) is str:
            if query == "All":
                if "_value" in self.data[id] and len(self.data[id]) == 1:
                    return self.data[id]["_value"]
                output = self.data[id]
                return output
            if query in self.data[id]:
                output[query] = self.data[id][query]
                return output
            raise KeyError("Unknown field must be in {0}".format(self.data[id].keys()))
        
        try:
            iterator = iter(query)
        except TypeError:
            if query == "All":
                output = self.data[id]
                return output
            if query in self.data[id]:
                output[query] = self.data[id][query]
                return output
            raise KeyError("Unknown field must be in {0}".format(self.data[id].keys()))
        
        for field in iterator:
            if field in self.data[id]:
                output[field] = self.data[id][field]
            else:
                raise KeyError("Unknown field must be in {0}".format(self.data[id].keys()))
        return output
    
    def get_mutliple_item(self, id_list, query="All"):
        """Return all entries with ids in `id_list` of the dataset with fields in `query` """
        output = {}
        for id in id_list:
            output[id] = self.get_item(id, query)

    def get_all_item(self, query="All"):
        """Get all entry of the dataset with fields precised in `query`"""
        output = {}
        for id in self.data.keys():
            output[id] = self.get_item(id, query)
        return output

    def aggregate(self, query):
        """Aggregate the data of same type as in `query` for all the dataset"""
        dtg = DatasetGroup("aggregated " + str(self.name))
        data = self.get_all_item(query)
        for id in self.data.keys():
            sub_data = data[id]
            for key, value in sub_data.items():
                if not dtg.is_in_group(key):
                    dt = Dataset(key, self.niter)
                    dt = dtg.add_dataset(dt)
                else:
                    dt = dtg.get_sub(key)
                dt.add(id, {"_value": value})
        return dtg

    def group_by(self, selector):
        """Regroup data in dataset by value of the field `Selector` """
        dtg = DatasetGroup(selector)
        for i in self.data.keys():
            if selector in self.data[i]:
                if not dtg.is_in_group(self.data[i][selector]):
                    dt = Dataset(self.data[i][selector], self.niter)
                    dt = dtg.add_dataset(dt)
                else:
                    dt = dtg.get_sub(self.data[i][selector])
                dt.add(i, self.get_item(i))
        return dtg

    def copy(self):
        """Copy object"""
        copyds = Dataset(self.name, self.niter)
        for i in self.data.keys():
            copyds.add(i, self.get_item(i).copy())
        return copyds
    
    def keys(self):
        """Return possible keys of the structure"""
        return self.data.keys()

    def __str__(self):
        return "Dataset '" + str(self.name) + "' with data: " + 10 * "-" + "\n" + str(self.data)


class DatasetGroup:

    def __init__(self, name):
        self.name = str(name)
        self.size = 0
        self.subs = {}
    
    def add_dataset(self, dataset):
        """Add a Dataset object `dataset` in subs"""
        self.size += 1
        if dataset.name in self.subs:
            print("WARNING: Replacing dataset might be a name problem")
        self.subs[dataset.name] = dataset
        return dataset
    
    def add_datasetgroup(self, name):
        """Add a DatasetGroup object with name `name` in subs"""
        datagroup = DatasetGroup(name)
        self.subs[name] = datagroup
        return datagroup
    
    def get_sub(self, name):
        """Return the sub with name `name` if in subs"""
        if name in self.subs:
            return self.subs[name]
        raise KeyError("The dataset with name {} is not contained in the group".format(name))

    def is_in_group(self, name):
        """Test is subname `name` is in group"""
        return name in self.subs
    
    def get_item(self, query):
        """Get item(s) matching `query`
        if `query` is iterable will return a DatasetGroup with all subs matched"""
        if type(query) is str:
            if self.is_in_group(query):
                return self.get_sub(query)
            raise KeyError("`query` not valid: not a dataset in the collection")

        try:
            iterator = iter(query)
        except TypeError:
            if self.is_in_group(query):
                return self.get_sub(query)
            raise KeyError("`query` not valid: not a dataset in the collection")
        
        output = DatasetGroup(self.name + str(query))
        for field in iterator:
            if self.is_in_group(field):
                output.add_dataset(self.get_sub(field).copy())
            else:
                raise KeyError("Unknown field")
        return output
    
    def get_all_item(self):
        """Get all subs that are in the DatasetGroup"""
        return self
    
    def group_by(self, selector):
        """Group by each content with selector"""
        copy_dtg = self.copy()
        for name, content in copy_dtg.subs.items():
            copy_dtg.subs[name] = content.group_by(selector)
        return copy_dtg
    
    def aggregate(self, query):
        """Aggregate each content with query"""
        copy_dtg = self.copy()
        for name, content in copy_dtg.subs.items():
            copy_dtg.subs[name] = content.aggregate(query)
        return copy_dtg
    
    def copy(self):
        """Deep copy the object"""
        copy_dtg = DatasetGroup(self.name)
        copy_dtg.size = self.size
        for name, content in self.subs.items():
            copy_dtg.subs[name] = content.copy()
        return copy_dtg
    
    def keys(self):
        """Return possible keys of the DatasetGroup"""
        return self.subs.keys()

    def __str__(self):
        string = "DatasetGroup '" + self.name + "' " + 10 * "-" + "\n"
        for name in self.subs.keys():
            string += str(name) + "\n"
            string += str(self.subs[name]) + "\n"
        return string
    
################################################################################################
# Dataset measure
################################################################################################

def measure_frequency_diadic_pattern(link_adjacency_matrix, phenotype_table, niter):
    """Measure the frequency of each diadic pattern"""
    pattern_freq = Dataset("Diadic", niter)
    size = link_adjacency_matrix.shape[0]
    for i in range(size):
        pattern_freq.add(i, {".": 0, "->": 0, "<-": 0,"--": 0, "Phenotype": phenotype_table[i]})
    for i in range(size):
        for j in range(i+1, size):
            if link_adjacency_matrix[i, j] == link_adjacency_matrix[j, i]:
                if link_adjacency_matrix[i, j] > 0:
                    pattern_freq.modify_field_in_id(i, "--", 1, mode="+")
                    pattern_freq.modify_field_in_id(j, "--", 1, mode="+")
                else:
                    pattern_freq.modify_field_in_id(i, ".", 1, mode="+")
                    pattern_freq.modify_field_in_id(j, ".", 1, mode="+")
            else:
                if link_adjacency_matrix[i, j] > 0:
                    pattern_freq.modify_field_in_id(i, "->", 1, mode="+")
                    pattern_freq.modify_field_in_id(j, "<-", 1, mode="+")
                else:
                    pattern_freq.modify_field_in_id(j, "->", 1, mode="+")
                    pattern_freq.modify_field_in_id(i, "<-", 1, mode="+")
    return pattern_freq

triadic_skeleton_global = {
    "000000": {"Number": 0, "Type": "Diadic", "Transitive": False},
    "001010": {"Number": 0, "Type": "Diadic", "Transitive": False},
    "011011": {"Number": 0, "Type": "Diadic", "Transitive": False},
    "002110": {"Number": 0, "Type": "Triadic", "Transitive": False},
    "110002": {"Number": 0, "Type": "Triadic", "Transitive": False},
    "011101": {"Number": 0, "Type": "Triadic", "Transitive": False},
    "012210": {"Number": 0, "Type": "Triadic", "Transitive": True},
    "111111": {"Number": 0, "Type": "Triadic", "Transitive": False},
    "012111": {"Number": 0, "Type": "Triadic", "Transitive": False},
    "112121": {"Number": 0, "Type": "Triadic", "Transitive": False},
    "022211": {"Number": 0, "Type": "Triadic", "Transitive": True},
    "211022": {"Number": 0, "Type": "Triadic", "Transitive": False},
    "111012": {"Number": 0, "Type": "Triadic", "Transitive": False},
    "112112": {"Number": 0, "Type": "Triadic", "Transitive": False},
    "122212": {"Number": 0, "Type": "Triadic", "Transitive": True},
    "222222": {"Number": 0, "Type": "Triadic", "Transitive": True}
}

def complex_degree(i, j, k, link_adjacency):
    """Return complex degree of `i` in the triangle drawn by (i,j,k)"""
    out_deg = link_adjacency[i, j] + link_adjacency[i, k]
    in_deg = link_adjacency[j, i] + link_adjacency[k, i]
    return out_deg, in_deg

def triadic_order(tuple):
    """For recognizing triadic pattern using tuple sort"""
    return tuple[0] + tuple[1], tuple[0]

def get_id_from_degrees(deg_seq):
    """Get the id of the triadic pattern"""
    id = ""
    for i in range(2):
        for j in range(3):
            id += str(deg_seq[j][i])
    return id

def measure_global_frequency_triadic_pattern(link_adjacency_matrix, niter):
    """Measure the frequency of each triadic pattern"""
    pattern_freq = Dataset("Triadic", niter)
    for key, dt_dict in triadic_skeleton_global.items():
        pattern_freq.add(key, dt_dict.copy())
    size = link_adjacency_matrix.shape[0]
    for i in range(size):
        for j in range(i+1, size):
            for k in range(j+1, size):
                deg_seq = []
                deg_seq.append(complex_degree(i, j, k, link_adjacency_matrix))
                deg_seq.append(complex_degree(j, i, k, link_adjacency_matrix))
                deg_seq.append(complex_degree(k, j, i, link_adjacency_matrix))
                triangle_id = get_id_from_degrees(sorted(deg_seq, key=triadic_order))
                pattern_freq.modify_field_in_id(triangle_id, "Number", 1, mode="+")
    return pattern_freq
