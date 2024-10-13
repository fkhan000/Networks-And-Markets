
# CS5854 HW2_p9.py
# IMPORTANT: Please read all comments!

# ==========Collaborators=============
# Please enter here the netids of all members of your group (yourself included.)
# Groups are at maximum 3 people.
# You may submit the same python code, or you may choose to each submit 
# your own code on gradescope (in which case your submission will be 
# graded separately). But as long as you collaborate -- even a little bit -- 
# please put your collaborator's netids here so that we can track groups.
# ====================================
authors = ['netID1','netID2']

# ==========Python Version============
# Which version of python are you using? 
# "Python 2" or "Python 3"? (Autograder defaults to 3) 
# ====================================
python_version = "YourPythonVersionHere"


# ======Submission and Packages=======
# Make sure to submit hw1.py to Gradescope, and make
# sure to keep the name the same. If your code spans
# multiple files, upload all of them. You do not
# need to submit facebook_combined.txt. 
# Our autograder only supports the following external
# packages:
# [numpy, pandas, scipy, matplotlib, random]
# please contact us before submission if you want another 
# package approved, if reasonable.
# ====================================
import numpy as np
import matplotlib.pyplot as plt
from hw2_p8 import UndirectedGraph


# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures 
# (i.e. the arguments and return types each function takes). 
# But feel free to add other methods and attributes as needed. 
# We will pass your grade through an autograder which expects a specific 
# interface. Please do re-use code from prior HWs where appropriate.
# =====================================
class WeightedDirectedGraph:
    def __init__(self,number_of_nodes):
        '''Assume that nodes are represented by indices/integers between 0 and
        number_of_nodes - 1.'''
        # TODO: Implement this method
        pass
    
    def set_edge(self, origin_node, destination_node, weight=1):
        ''' Modifies the weight for the specified directed edge, from origin 
        to destination node, with specified weight (an integer >= 0). 
        If weight = 0, removes the edge from the graph. If edge previously wasn't 
        in the graph, adds a new edge with specified weight. The graph should
        support self-loops.'''
        # TODO: Implement this method
        pass
    
    def edges_from(self, origin_node):
        ''' This method shold return a list of all the nodes destination_node 
        such that there is a directed edge (origin_node, destination_node) in the 
        graph (i.e. with weight > 0), supporting self-loops.'''
        # TODO: Implement this method
        pass
    
    def get_edge(self, origin_node, destination_node):
        ''' This method should return the weight (an integer > 0) 
            if there is an edge between origin_node and 
            destination_node, and 0 otherwise.'''
        # TODO: Implement this method
        pass
    
    def number_of_nodes(self):
        ''' This method should return the number of nodes in the graph'''
        # TODO: Implement this method
        pass

# === Problem 9(a) ===
def max_flow(G, s, t):
    '''Given a WeightedDirectedGraph G, a source node s, a destination node t,
       compute the (integer) maximum flow from s to t, treating the weights of G 
       as capacities. Return a tuple (v, F) where v is the integer value of the flow,
       and F is a maximum flow for G, represented by another WeightedDirectedGraph
       where edge weights represent the final allocated flow along that edge.'''
    # TODO: Implement this method
    pass

# === Problem 9(c) ===
def max_matching(n, m, C):
    '''Given n drivers, m riders, and a set of matching constraints C,
    output a maximum matching. Specifically, C is a n x m array, where
    C[i][j] = 1 if driver i (in 0...n-1) and rider j (in 0...m-1) are compatible.
    If driver i and rider j are incompatible, then C[i][j] = 0. 
    Return an n-element array M where M[i] = j if driver i is matched with rider j,
    and M[i] = None if driver i is not matched.'''
    # TODO: Implement this method
    pass

# === Problem 9(d) ===
def random_driver_rider_bipartite_graph(n, p):
    '''Returns an n x n constraints array C as defined for max_matching, representing 
    a bipartite graph with 2n nodes, where each vertex in the left half is connected 
    to any given vertex in the right half with probability p.'''
    # TODO: Implement this method
    pass

def main():
    # TODO: Put your analysis and plotting code here for 9(d)
    print("hello world")

if __name__ == "__main__":
    main()
