
# CS5854 HW1.py
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
from collections import deque


# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures 
# (i.e. the arguments and return types each function takes). 
# But feel free to add other methods and attributes as needed. 
# We will pass your grade through an autograder which expects a specific 
# interface.
# =====================================
class UndirectedGraph:
    def __init__(self,number_of_nodes):
        '''Assume that nodes are represented by indices/integers between 0 and number_of_nodes - 1.'''
        self.nodes = [set() for _ in range(number_of_nodes)]
    
    def add_edge(self, nodeA, nodeB):
        self.nodes[nodeA].add(nodeB)
        self.nodes[nodeB].add(nodeA)
    
    def edges_from(self, nodeA):
        return list(self.nodes[nodeA])
    
    def check_edge(self, nodeA, nodeB):
        return nodeA in self.nodes[nodeB]
    
    def number_of_nodes(self):
        return len(self.nodes)



# Problem 9(a)
def create_graph(n,p):

    ''' Given number of nodes n and probability p, output an UndirectedGraph with n nodes, where each
    pair of nodes is connected by an edge with probability p'''
    G = UndirectedGraph(n)

    adjency_matrix = np.random.rand(n, n) < p
    for row in range(len(adjency_matrix)):
             for col in range(len(adjency_matrix)):
                  if adjency_matrix[row][col]:
                      G.add_edge(row, col)
    return G

# Problem 9(b)
def shortest_path(G,i,j):
    ''' Given an UndirectedGraph G and nodes i,j, output the length of the shortest path between i and j in G.
    If i and j are disconnected, output -1.'''
    queues = [deque([(i, 0)]), deque([(j, 0)])]
    visited = [{i:0}, {j:0}]

    while queues[0] and queues[1]:
        
        for index in range(2):
            current_node, path_length = queues[index].popleft()

            if current_node in visited[(index + 1) % 2]:
                    return path_length + visited[(index + 1) % 2][current_node]

            for neighbor in G.edges_from(current_node):
                    if neighbor not in visited[index]:
                        visited[index][neighbor] = path_length + 1
                        queues[index].append((neighbor, path_length + 1))                    
    return -1

# Problem 9(c)
def avg_shortest_path(G, num_samples=1000):
    ''' Given an UndirectedGraph G, return an estimate of the average shortest path in G, where the average is taken
    over all pairs of CONNECTED nodes. The estimate should be taken by sampling num_samples random pairs of connected nodes, 
    and computing the average of their shortest paths. Return a decimal number.'''
    indices = list(range(len(G.nodes)))
    avg_length = 0

    for _ in range(num_samples):
        length = -1
        while(length == -1):
             i, j = np.random.choice(indices, 2, replace=False)
             length = shortest_path(G, i, j)
        avg_length += length

    return avg_length / num_samples

# Problem 9(d)
def simulate_varying_prob(prob_vals):
    avg_distances = []
    for p in prob_vals:
        G = create_graph(1000, p)
        avg_distance = avg_shortest_path(G)          
        avg_distances.append(avg_distance)

    plt.plot(prob_vals, avg_distances)
    plt.show()


# Problem 10(a)
def create_fb_graph(filename = "facebook_combined.txt"):
    ''' This method should return a undirected version of the facebook graph as an instance of the UndirectedGraph class.
    You may assume that the input graph has 4039 nodes.'''  
    G = UndirectedGraph(4039)

    for line in open(filename):
        nodeA, nodeB = line.split(" ")
        G.add_edge(int(nodeA), int(nodeB))
    return G

# Problem 10(b)
def average_fb_graph_length():
    G = create_fb_graph()
    return avg_shortest_path(G)

# Problem 10(c) if applicable.
def probability_connected_edge(G: UndirectedGraph, num_samples=1000):
    indices = list(range(len(G.nodes)))

    prob_connected = 0

    for _ in range(num_samples):
        i, j = np.random.choice(indices, 2, replace=False)
        if j in G.nodes[i]:
            prob_connected += 1
    return prob_connected / num_samples

     
def main():
    G = create_graph(1000, 0.1)
    import time
    start = time.time()
    print(avg_shortest_path(G))
    print(time.time() - start)

if __name__ == "__main__":
    main()
