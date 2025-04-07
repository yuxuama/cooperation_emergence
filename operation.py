"""operation.py
Implement the operation stack structure in order to save
"""

import pandas as pd
import numpy as np
import h5py
import os

class OperationStack:

    def __init__(self, stack_dir=None):
        """If no `stack_file` is given return empty stack else load the stacks from the file
        if `init_trust` and `init_link` are given they must be np.ndarray
        """
        self.iter_number = 0
        self.read_pointer = 0
        self.read_interaction = 0
        
        self.activated = True # When True the operation stacks can be modified otherwise can only be read
        # Defining stacks
        # Format:
        # Interaction number: the number of the interaction when the operation takes place
        # Operation name: Either "Link" or "Trust": specify on which network we act
        # i: the start vertex
        # j: the end vertex
        # value: increment value in the network
        self.stacks = {
            "Interaction number": [],
            "Operation name": [],
            "i": [],
            "j": [],
            "Value": [] 
        }

        # Init matrices
        self.trust = None
        self.link = None

        if stack_dir is not None:
            self.load_stack_from_dir(stack_dir)

    def add_link(self, i, j):
        """Add operation corresponding to adding a link to the stack"""
        if self.activated:
            self.stacks["Interaction number"].append(self.iter_number)
            self.stacks["Operation name"].append("Link")
            self.stacks["i"].append(i)
            self.stacks["j"].append(j)
            self.stacks["Value"].append(1)

    def remove_link(self, i, j):
        """Add operation corresponding to removing a link to the stack"""
        if self.activated:
            self.stacks["Interaction number"].append(self.iter_number)
            self.stacks["Operation name"].append("Link")
            self.stacks["i"].append(i)
            self.stacks["j"].append(j)
            self.stacks["Value"].append(-1)

    def increment_trust(self, i, j, increment):
        """Add operation corresponding to a trust value update"""
        if self.activated:
            self.stacks["Interaction number"].append(self.iter_number)
            self.stacks["Operation name"].append("Trust")
            self.stacks["i"].append(i)
            self.stacks["j"].append(j)
            self.stacks["Value"].append(increment)

    def next_iter(self):
        """Notify the stack the interaction is not the same as before"""
        if self.activated:
            self.iter_number += 1

    def save(self, dir_path):
        """Save the stack in a folder"""
        os.makedirs(dir_path, exist_ok=True)       
        df = pd.DataFrame(self.stacks)
        df.to_csv(dir_path + "stack.csv", index=False)

        if self.trust is not None and self.link is not None:
            f = h5py.File(dir_path + "init.h5", 'w')
            f["Trust"] = self.trust
            f["Link"] = self.link

    def load_stack_from_dir(self, dir_path):
        """Set all the stack according to the csv `stack_file`
        Warning: overwrites all existing values
        """
        df = pd.read_csv(dir_path + "stack.csv")
        self.stacks = df.to_dict('list')
        self.set_matrices_from_file(dir_path + "init.h5")
        self.iter_number = self.stacks["Interaction number"][-1]
    
    def set_matrices_from_file(self, filepath):
        """set the link and trust adjacency matrices stored in the hdf5 file with path `filepath`"""
        f = h5py.File(filepath, 'r')
        self.link = np.array(f.get("Link"), dtype=int)
        self.trust = np.array(f.get("Trust"))

    def set_link_from_array(self, link_array):
        """set the link adjacency matrix"""
        self.link = link_array

    def set_trust_from_array(self, trust_array):
        """set the trust adjacency matrix"""
        self.trust = trust_array
    
    def resolve_one(self):
        """Read all operation during an interaction and update matrices"""
        assert self.link is not None and self.trust is not None
        assert self.read_interaction <= self.iter_number
        iter_stack = self.stacks["Interaction number"]

        while self.read_pointer < len(iter_stack) and iter_stack[self.read_pointer] == self.read_interaction:
            name = self.stacks["Operation name"][self.read_pointer]
            i = self.stacks["i"][self.read_pointer]
            j = self.stacks["j"][self.read_pointer]
            value = self.stacks["Value"][self.read_pointer]
            
            if name == "Link":
                self.link[i, j] += value
            elif name == "Trust":
                self.trust[i, j] += value
            
            self.read_pointer += 1

        self.read_interaction += 1
        return self.trust, self.link

    def amend_one(self):
        """Amend all operation during of the current interaction and update matrices"""
        assert self.link is not None and self.trust is not None
        assert self.read_interaction > 0
        iter_stack = self.stacks["Interaction number"]
        self.read_pointer -= 1
        self.read_interaction -= 1
        while self.read_pointer > -1 and iter_stack[self.read_pointer] == self.read_interaction:
            name = self.stacks["Operation name"][self.read_pointer]
            i = self.stacks["i"][self.read_pointer]
            j = self.stacks["j"][self.read_pointer]
            value = self.stacks["Value"][self.read_pointer]
            
            if name == "Link":
                self.link[i, j] -= value
            elif name == "Trust":
                self.trust[i, j] -= value
            
            self.read_pointer -= 1

        self.read_pointer += 1
        return self.trust, self.link
    
    def resolve(self, inter_number):
        """Return the state of trust and link in this order at the `inter_number`-th interaction
        Note that `inter_number` must be strictly positive (natural counting for humans). A special case
        is added with -1 which gives the final states."""
        if inter_number > self.iter_number+1:
            raise "More interaction requested than simulated"
        if inter_number == -1:
            inter_number = self.iter_number+1

        if inter_number >= self.read_interaction:
            for _ in range(inter_number - self.read_interaction):
                self.resolve_one()
        else:
            for _ in range(self.read_interaction - inter_number):
                self.amend_one()
        
        return self.trust, self.link

    def __len__(self):
        return self.iter_number