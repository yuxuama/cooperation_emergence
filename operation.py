"""operation.py
Implement the operation stack structure in order to save
"""

import pandas as pd
import numpy as np

class OperationStack:

    def __init__(self, init_trust=None, init_link=None, stack_file=None):
        """If no `stack_file` is given return empty stack else load the stacks from the file
        if `init_trust` and `init_link` are given they must be np.ndarray
        """
        self.iter_number = 0
        self.read_pointer = 0
        self.read_interaction = 0

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

        if stack_file is not None:
            self.load_stack_from_file(stack_file)

        # Init matrices
        self.trust = init_trust
        self.link = init_link

    def add_link(self, i, j):
        """Add operation corresponding to adding a link to the stack"""
        self.stacks["Interaction number"].append(self.iter_number)
        self.stacks["Operation name"].append("Link")
        self.stacks["i"].append(i)
        self.stacks["j"].append(j)
        self.stacks["Value"].append(1)

    def remove_link(self, i, j):
        """Add operation corresponding to removing a link to the stack"""
        self.stacks["Interaction number"].append(self.iter_number)
        self.stacks["Operation name"].append("Link")
        self.stacks["i"].append(i)
        self.stacks["j"].append(j)
        self.stacks["Value"].append(-1)

    def increment_trust(self, i, j, increment):
        """Add operation corresponding to a trust value update"""
        self.stacks["Interaction number"].append(self.iter_number)
        self.stacks["Operation name"].append("Trust")
        self.stacks["i"].append(i)
        self.stacks["j"].append(j)
        self.stacks["Value"].append(increment)

    def next_iter(self):
        """Notify the stack the interaction is not the same as before"""
        self.iter_number += 1

    def save(self, save_file):
        """Save the stack in a file"""        
        df = pd.DataFrame(self.stacks)
        df.to_csv(save_file, index=False)

    def load_stack_from_file(self, stack_file):
        """Set all the stack according to the csv `stack_file`
        Warning: overwrites all existing values
        """
        df = pd.read_csv(stack_file)
        self.stacks = df.to_dict('list')
        self.iter_number = self.stacks["Interaction number"][-1]
    
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

        self.read_interaction += 1 # This work because at each interaction an operation occurs but not safe in theory
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
