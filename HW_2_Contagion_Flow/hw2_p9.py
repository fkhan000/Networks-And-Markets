
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

def bfs(G, source , destination, parent):
    """Returns True if there is a path from source and destination in graph
    Also fills 'parent' to store the path."""
    visited = [False] * G.number_of_nodes()
    queue = Queue()
    queue.append(source)
    visited[source] = True
    
    # while queue is not empty
    while queue:
        # get the top node on the queue
        current_node = queue.pop_front()
        # we add neighbors of current node to queue
        for neighbor in G.edges_from(current_node):
            if not visited[neighbor]:
                queue.append(neighbor)
                # set visited flag
                visited[neighbor] = True
                #and then set neighbor's parent to be current node
                parent[neighbor] = current_node
                #if we reached the destination node then we terminate and return true
                if  neighbor == destination:
                    return True
    # if we reached here then there is no path from source to destination
    return False

# === Problem 9(a) ===
def max_flow(G, s, t):
    '''Given a WeightedDirectedGraph G, a source node s, a destination node t,
       compute the (integer) maximum flow from s to t, treating the weights of G 
       as capacities. Return a tuple (v, F) where v is the integer value of the flow,
       and F is a maximum flow for G, represented by another WeightedDirectedGraph
       where edge weights represent the final allocated flow along that edge.'''
    
    # parent list, will be used to keep track of path from source to sink found by bfs
    parent = [-1] * G.number_of_nodes()
    maximum_flow = 0

    # we create our residual graph which initially will just be a copy of G
    RG = WeightedDirectedGraph(G.number_of_nodes())
    for i in range(G.number_of_nodes()):
        for j in G.edges_from(i):
            cap = G.get_edge(i, j)
            RG.set_edge(i, j, cap)

    # Then we augment the flow while there is still a path from source to sink
    while bfs(RG, s, t, parent):
        path_flow = float('Inf')
        v = t

        # We walk backwards in the path and find the edge in our path with the smallest capacity
        # this will be maximum flow we can push through this path 
        while v != s:
            u = parent[v]
            path_flow = min(path_flow, RG.get_edge(u, v))
            v = parent[v]

        # We then update the capacities of the edges and reverse edges in the residual graph
        # (residual edges increase in capacity and normal edges decrease)
        v = t
        while v != s:
            u = parent[v]
            current_cap = RG.get_edge(u, v)
            RG.set_edge(u, v, current_cap - path_flow)
            current_cap = RG.get_edge(v, u)
            RG.set_edge(v, u, current_cap + path_flow)
            v = parent[v]
        # Finally we update maximum flow
        maximum_flow += path_flow
    
    # To get flow graph we simply subtract the capacities of corresponding normal edges in the original
    # and residual graph
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
    
    # this matching problem can be considered as a special case of max flow problem
    # where we add source and destination node to left and right side of graph respectively

    # we weighted directed graph representing the current matching problem
    match_graph = WeightedDirectedGraph(n + m + 2)
    for i in range(n):
        for j in range(n, m + n):
            match_graph.set_edge(i, j, C[i][j-n])
    source = n + m
    sink = n + m + 1

    # then we add our source and sink nodes to left and right side of graph respectively
    for i in range(n):
        match_graph.set_edge(source, i, 1)
    for j in range(n, n + m):
        match_graph.set_edge(j, sink, 1)
    
    # we try to push as much flow as possible from source to sink and in doing so
    # solve matching problem
    _, flow_graph = max_flow(match_graph, source, sink)

    M = [None for _ in range(n)]
    # To find assigned matching in flow graph
    # we check to see which edges between the drivers and riders have a flow of 1 in the graph 
    for i in range(n):
        for j in range(n, n + m):
            if flow_graph.cap_matrix[i][j] == 1:
                M[i] = j - n
                break
    return M

# === Problem 9(d) ===
def random_driver_rider_bipartite_graph(n, p):
    '''Returns an n x n constraints array C as defined for max_matching, representing 
    a bipartite graph with 2n nodes, where each vertex in the left half is connected 
    to any given vertex in the right half with probability p.'''

    # zero out the constraint array
    constraint_arr = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            # and with probability p add an edge from the left half of graph to the right
            if np.random.random() <= p:
                constraint_arr[i][j] = 1
    return constraint_arr

def simulate_matching(n, p, num_trials = 100):
    '''Experimentally derives the probability of a complete match occurring in a graph with 2n in which
    edges are generated in the bipartite graph with probability p'''

    prob_match = 0

    for _ in range(num_trials):
        # generate the constraint array
        constraint_arr = random_driver_rider_bipartite_graph(n, p)
        # find the max matching
        matches = max_matching(n, n, constraint_arr)
        # and check to see if a complete match was made
        if sum([1 if match is not None else 0 for match in matches]) == n:
            prob_match += 1
    prob_match /= num_trials
    return prob_match

def test_fig_6_1():
    G = WeightedDirectedGraph(4)
    G.set_edge(0, 1, 12)
    G.set_edge(0, 2, 39)
    G.set_edge(1, 2, 23)
    G.set_edge(2, 1, 11)
    G.set_edge(1, 3, 23)
    G.set_edge(2, 3, 27)
    maximum_flow, flow_graph = max_flow(G, 0, 3)
    assert maximum_flow == 50
    assert flow_graph.get_edge(0, 1) == 12
    assert flow_graph.get_edge(0, 2) == 38
    assert flow_graph.get_edge(1, 2) == 0
    assert flow_graph.get_edge(2, 1) == 11
    assert flow_graph.get_edge(2, 3) == 27
    assert flow_graph.get_edge(1, 3) == 23

def test_fig_6_3():
    G = WeightedDirectedGraph(4)
    G.set_edge(0, 1, 12)
    G.set_edge(0, 2, 39)
    G.set_edge(1, 2, 23)
    G.set_edge(1, 3, 9)
    G.set_edge(2, 3, 15)
    maximum_flow, flow_graph = max_flow(G, 0, 3)

    assert maximum_flow == 24
    assert flow_graph.get_edge(0, 1) == 9
    assert flow_graph.get_edge(0, 2) == 15
    assert flow_graph.get_edge(2, 3) == 15
    assert flow_graph.get_edge(1, 3) == 9

def test_max_matching_1():
    constraint_array = np.ones((5, 5))
    for i in range(1, 5):
        constraint_array[0][i] = 0
    M = max_matching(5, 5, constraint_array)
    assert M[0] == 0
    num_matches = sum([1 if match is not None else 0 for match in M])
    assert num_matches == 5

def test_max_matching_2():
    constraint_array = np.zeros((5, 5))
    for i in range(1, 5):
        constraint_array[0][i] = 1
    constraint_array[1][1] = 1
    constraint_array[1][2] = 1
    constraint_array[2][1] = 1
    constraint_array[3][2] = 1
    constraint_array[4][3] = 1

    M = max_matching(5, 5, constraint_array)
    num_matches = sum([1 if match is not None else 0 for match in M])

    assert num_matches == 4
    assert M[0] == 4

def main():
    # === Problem 9(a) === #
    test_fig_6_1()
    test_fig_6_3()
    print("Max Flow Implementation successfully passed test cases!")

    # === Problem 9(c) === #
    test_max_matching_1()
    test_max_matching_2()
    print("Max Matching Implementation successfully passed test cases!")

    # === Problem 9(d) === #
    N = 100
    probs_match = []
    # we find the probability of a complete match for varying values of p
    for prob in np.arange(0.1, 1, 0.05):
        probs_match.append(simulate_matching(N, prob))
    
    #and then plot the probability of a complete match against the probability of compatibility
    plt.plot(np.arange(0.1, 1, 0.05), probs_match)
    plt.title("Probability of Complete Match Against Probabilty of Compatibility")
    plt.xlabel("Probability of Compatibility")
    plt.ylabel("Probability of Complete Match")

if __name__ == "__main__":
    main()
