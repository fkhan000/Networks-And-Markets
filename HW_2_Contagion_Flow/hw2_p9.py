
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
authors = ['fsk36']

# ==========Python Version============
# Which version of python are you using? 
# "Python 2" or "Python 3"? (Autograder defaults to 3) 
# ====================================
python_version = "3.11.5"


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
        self.nodes = [set() for _ in range(number_of_nodes)]
        self.cap_matrix = np.ones((len(self.nodes), len(self.nodes)))*0
    
    def set_edge(self, origin_node, destination_node, weight=1):
        ''' Modifies the weight for the specified directed edge, from origin 
        to destination node, with specified weight (an integer >= 0). 
        If weight = 0, removes the edge from the graph. If edge previously wasn't 
        in the graph, adds a new edge with specified weight. The graph should
        support self-loops.'''

        if weight == 0:
            self.nodes[origin_node].discard(destination_node)
            self.cap_matrix[origin_node][destination_node] = 0
        else:
            if destination_node not in self.nodes[origin_node]:
                self.nodes[origin_node].add(destination_node)
            self.cap_matrix[origin_node][destination_node] = weight

    def edges_from(self, origin_node):
        ''' This method shold return a list of all the nodes destination_node 
        such that there is a directed edge (origin_node, destination_node) in the 
        graph (i.e. with weight > 0), supporting self-loops.'''
        return list(self.nodes[origin_node])
    
    def get_edge(self, origin_node, destination_node):
        ''' This method should return the weight (an integer > 0) 
            if there is an edge between origin_node and 
            destination_node, and 0 otherwise.'''
        return self.cap_matrix[origin_node][destination_node]
    
    def number_of_nodes(self):
        ''' This method should return the number of nodes in the graph'''
        return len(self.nodes)

class Node:
    """Base unit of linked list, stores previous and next node as well as payload."""
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class Queue:
    """The Queue class. Supports appending to end and popping from beginning of linked list."""
    def __init__(self):
        """To make appending and popping O(1) operations, we store both start and end nodes."""
        self.start = None
        self.end = None
        self.length = 0

    def append(self, data):
        """Appends a node with payload data to end of queue."""
        # we create a new node
        node = Node(data)
        # if the queue is empty
        if not self.start:
            # we assign the start and end to this node
            self.start = self.end = node
        else:
            # else we append it to the end by making the next for self.end the node
            self.end.next = node
            # having the node point backwards to the end node
            node.prev = self.end
            # and then making our end node the new node inserted
            self.end = node
        # finally we increment the length
        self.length += 1
    
    def pop_front(self):
        """Removes a node from beginning of queue and returns the payload."""

        # we store the start node and assign the new start to be the node next to it
        node = self.start
        self.start = self.start.next
        
        # if there is another node remaining
        if self.start:
            # have it point backwards to None
            self.start.prev = None
        else:
            # else the queue is now empty and so we make the end node none as well
            self.end = None

        # we decrement the length and return the payload
        self.length -= 1
        return node.data

    def __len__(self):
        return self.length

from collections import deque

def bfs(RG, s, t, parent):
    """Returns True if there is a path from source 's' to sink 't' in residual graph.
    Also fills 'parent' to store the path."""
    visited = [False] * RG.number_of_nodes()
    queue = deque([s])
    visited[s] = True
    
    while queue:
        u = queue.popleft()

        for v in RG.edges_from(u):
            # Check for positive residual capacity
            if not visited[v]:
                queue.append(v)
                visited[v] = True
                parent[v] = u
                if v == t:
                    return True
    return False

# === Problem 9(a) ===
def max_flow(G, s, t):
    '''Given a WeightedDirectedGraph G, a source node s, a destination node t,
       compute the (integer) maximum flow from s to t, treating the weights of G 
       as capacities. Return a tuple (v, F) where v is the integer value of the flow,
       and F is a maximum flow for G, represented by another WeightedDirectedGraph
       where edge weights represent the final allocated flow along that edge.'''
    parent = [-1] * G.number_of_nodes()
    maximum_flow = 0

    # Create residual graph
    RG = WeightedDirectedGraph(G.number_of_nodes())
    for i in range(G.number_of_nodes()):
        for j in G.edges_from(i):
            cap = G.get_edge(i, j)
            RG.set_edge(i, j, cap)

    # Augment the flow while there is a path from source to sink
    while bfs(RG, s, t, parent):
        path_flow = float('Inf')
        v = t

        # Find the minimum residual capacity of the edges along the path filled by BFS
        while v != s:
            u = parent[v]
            path_flow = min(path_flow, RG.get_edge(u, v))
            v = parent[v]

        # Update residual capacities of the edges and reverse edges along the path
        v = t
        while v != s:
            u = parent[v]
            current_cap = RG.get_edge(u, v)
            RG.set_edge(u, v, current_cap - path_flow)
            current_cap = RG.get_edge(v, u)
            RG.set_edge(v, u, current_cap + path_flow)
            v = parent[v]

        maximum_flow += path_flow
    flow_graph = WeightedDirectedGraph(G.number_of_nodes())
    for u in range(G.number_of_nodes()):
        for v in G.edges_from(u):
            original_capacity = G.get_edge(u, v)
            remaining_capacity = RG.get_edge(u, v)
            flow = original_capacity - remaining_capacity
            if flow > 0:
                flow_graph.set_edge(u, v, flow)
    return maximum_flow, flow_graph


# === Problem 9(c) ===
def max_matching(n, m, C):
    '''Given n drivers, m riders, and a set of matching constraints C,
    output a maximum matching. Specifically, C is a n x m array, where
    C[i][j] = 1 if driver i (in 0...n-1) and rider j (in 0...m-1) are compatible.
    If driver i and rider j are incompatible, then C[i][j] = 0. 
    Return an n-element array M where M[i] = j if driver i is matched with rider j,
    and M[i] = None if driver i is not matched.'''
    
    match_graph = WeightedDirectedGraph(n + m + 2)
    for i in range(n):
        for j in range(n, m + n):
            match_graph.set_edge(i, j, C[i][j-n])
    source = n + m
    sink = n + m + 1

    for i in range(n):
        match_graph.set_edge(source, i, 1)
    for j in range(n, n + m):
        match_graph.set_edge(j, sink, 1)
    _, flow_graph = max_flow(match_graph, source, sink)

    M = [None for _ in range(n)]
    for i in range(n):
        for j in range(n, n + m):
            if flow_graph.cap_matrix[i][j] > 0:
                M[i] = j - n
                break
    return M

# === Problem 9(d) ===
def random_driver_rider_bipartite_graph(n, p):
    '''Returns an n x n constraints array C as defined for max_matching, representing 
    a bipartite graph with 2n nodes, where each vertex in the left half is connected 
    to any given vertex in the right half with probability p.'''
    constraint_arr = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if np.random.random() <= p:
                constraint_arr[i][j] = 1
    return constraint_arr

def simulate_matching(n, p, num_trials = 100):
    prob_match = 0

    for _ in range(num_trials):
        constraint_arr = random_driver_rider_bipartite_graph(n, p)
        matches = max_matching(n, n, constraint_arr)
        if matches == [1 for _ in range(n)]:
            prob_match += 1
    prob_match /= num_trials
    return prob_match
def main():
    N = 100
    probs_match = []
    for prob in np.arange(0.1, 1, 0.05):
        probs_match.append(simulate_matching(N, prob))
    
    plt.plot(np.arange(0.1, 1, 0.05), probs_match)
    plt.title("Probability of Complete Match Against Probabilty of Compatibility")
    plt.xlabel("Probability of Compatibility")
    plt.ylabel("Probability of Complete Match")


if __name__ == "__main__":
    main()
