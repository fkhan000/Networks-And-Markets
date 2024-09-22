
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

    def __str__(self):
        matr = [[0 for _ in range(len(self.nodes))] for _ in range(len(self.nodes))]
        for row in range(len(self.nodes)):
            for col in range(len(self.nodes)):
                if self.check_edge(row, col):
                    matr[row][col] = 1
        return np.array(matr).__str__()

# In order to perform breadth first search, implemented queue as doubly linked list

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

    
# Problem 9(a)
def create_graph(n,p):

    ''' Given number of nodes n and probability p, output an UndirectedGraph with n nodes, where each
    pair of nodes is connected by an edge with probability p'''
    G = UndirectedGraph(n)
    
    adjency_matrix = np.random.rand(n, n) < p

    #since graph is undirected, we only look at upper triangular half of adjency matrix
    for row in range(len(adjency_matrix)):
             for col in range(row + 1, len(adjency_matrix[row])):
                  if adjency_matrix[row][col]:
                      G.add_edge(row, col)
    return G

# Problem 9(b)
def shortest_path(G,i,j):
    ''' Given an UndirectedGraph G and nodes i,j, output the length of the shortest path between i and j in G.
    If i and j are disconnected, output -1.'''

    # In order to find shortest path in unweighted and undirected graph, we perform BFS

    queue = Queue()
    # store node and path length from start node
    queue.append((i, 0))
    visited = [False] * G.number_of_nodes()
    visited[i] = True

    # while queue is not empty
    while len(queue):
        # get the top node on the queue
        current_node, path_length = queue.pop_front()
        # if we've reached end node
        if current_node == j:
            # we found shortest path so return path_length
            return path_length

        # we then add the neighbors of the current node to our queue
        for neighbor in G.edges_from(current_node):
            if not visited[neighbor]:
                # set visited flag
                visited[neighbor] = True
                queue.append((neighbor, path_length + 1))
    # if we reached here, there exists no path from i to j so we return -1        
    return -1

# Problem 9(c)
def avg_shortest_path(G, num_samples=1000):
    ''' Given an UndirectedGraph G, return an estimate of the average shortest path in G, where the average is taken
    over all pairs of CONNECTED nodes. The estimate should be taken by sampling num_samples random pairs of connected nodes, 
    and computing the average of their shortest paths. Return a decimal number.'''
    indices = list(range(G.number_of_nodes()))
    avg_length = 0

    for _ in range(num_samples):
        length = -1
        while(length == -1):
             i, j = np.random.choice(indices, 2, replace=False)
             length = shortest_path(G, i, j)
        avg_length += length

    return avg_length / num_samples

# Problem 9(d)

def is_connected(G: UndirectedGraph) -> bool:
    """Determines whether or not graph is connected by performing BFS from a node and checking if all
    all nodes in graph were visited"""

    # keep track of visited nodes
    visited = [False for _ in range(G.number_of_nodes())]
    visited_count = 0
    queue = Queue()
    queue.append(0)
    #queue = deque([0])

    while len(queue):
        # if we have visited all nodes no need to continue
        if visited_count == G.number_of_nodes():
            return True
        # get current node on top of queue
        #current_node = queue.popleft()
        current_node = queue.pop_front()

        if not visited[current_node]:
            visited[current_node] = True
            # increment visit count if this is new node
            visited_count += 1
        
        for neighbor in G.edges_from(current_node):
            # if new node
            if not visited[neighbor]:
                # add to queue
                queue.append(neighbor)

    return len(visited) == G.number_of_nodes()

def simulate_varying_prob(prob_vals):
    """Finds the average shortest path for randomly generated graphs with 1000 nodes
    and varying probabilities of edge generation and produces a plot."""

    avg_distances = []
    # for each probability
    for p in prob_vals:
        # construct the graph
        G = create_graph(1000, p)
        # if graph is disconnected recreate it
        while (not is_connected(G)):
            G = create_graph(1000, p)
        # get average shortest path
        avg_distance = avg_shortest_path(G)          
        avg_distances.append(avg_distance)

    # plot average shortest path against probability
    plt.plot(prob_vals, avg_distances)
    plt.xlabel("Probabilty of Edge Generation")
    plt.ylabel("Average Shortest Path")
    plt.title("Average Shortest Path vs Probability of Edge Generation")
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
    """Performs simulations to determine probability edge exists between two radnom nodes in given graph"""
    indices = list(range(len(G.nodes)))

    prob_connected = 0

    for _ in range(num_samples):
        # randomly select two nodes in graph without replacement
        i, j = np.random.choice(indices, 2, replace=False)
        # if j is a neighbor to i
        if j in G.nodes[i]:
            # increment the count
            prob_connected += 1

    return prob_connected / num_samples


def main():


    # 9C
    G = create_graph(1000, 0.1)
    avg_path_length = avg_shortest_path(G, num_samples=1000)

    print(f"The average path length for a randomly created graph with n = 1000 and p = 0.1 is {avg_path_length}\n")

    # 9D
    simulate_varying_prob(np.arange(0.01, 0.05, 0.01))
    simulate_varying_prob(np.arange(0.05, 1.05, 0.05))

    # 10b
    G = create_fb_graph()
    p = probability_connected_edge(G)
    print(f"The average shortest path for the facebook graph is {avg_shortest_path(G, num_samples=1000)}")

    # 10c
    print(f"The probability that two random nodes are connected is {p}")

    # 10d
    G_rand = create_graph(4039, p)
    avg_path_length = avg_shortest_path(G_rand, num_samples=1000)
    print(f"Meanwhile the average shortest path for a randomly generated graph with the same number of nodes and same probability of edge creation is {avg_path_length}")


if __name__ == "__main__":
    main()
