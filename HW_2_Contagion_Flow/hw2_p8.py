
# CS5854 HW2_p8.py
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
        '''Assume that nodes are represented by indices/integers between 0 
        and number_of_nodes - 1.'''
        self.nodes = [set() for _ in range(number_of_nodes)]
    
    def add_edge(self, nodeA, nodeB):
        ''' Adds an undirected edge to the graph, between nodeA and nodeB. 
        Order of arguments should not matter. Should support self-loops 
        (to be consistent with definitions in the textbook).'''
        self.nodes[nodeA].add(nodeB)
        self.nodes[nodeB].add(nodeA)
    
    def edges_from(self, nodeA):
        ''' This method shold return a list of all the nodes nodeB such that 
        nodeA and nodeB are connected by an edge (including nodeA if it has
        a self-loop).'''
        return list(self.nodes[nodeA])
    
    def check_edge(self, nodeA, nodeB):
        ''' This method should return true is there is an edge between 
        nodeA and nodeB, and false otherwise'''
        return nodeA in self.nodes[nodeB]
    
    def number_of_nodes(self):
        ''' This method should return the number of nodes in the graph'''
        return len(self.nodes)


def create_fb_graph(filename = "facebook_combined.txt"):
    ''' This method should return a undirected version of the facebook graph as 
    an instance of the UndirectedGraph class. You may assume that the input graph 
    has 4039 nodes.'''    
    G = UndirectedGraph(4039)

    for line in open(filename):
        nodeA, nodeB = line.split(" ")
        G.add_edge(int(nodeA), int(nodeB))
    return G


# === Problem 8(a) ===

def contagion_brd(G: UndirectedGraph, S, q):
    '''Given an UndirectedGraph G, a list of adopters S (a list of integers in [0, G.number_of_nodes - 1]),
       and a float threshold q, perform BRD as follows:
       - Permanently infect the nodes in S with X
       - Infect the rest of the nodes with Y
       - Run BRD on the set of nodes not in S.  A node should switch when the 
         fraction of infected neighbors is >= q.
       Return a list of all nodes infected with X after BRD converges.'''
    infected = set()
    for node in S:
        infected.add(node)
    terminate = False
    while(not terminate):
        terminate = True
        for i in range(G.number_of_nodes()):
            if i in infected:
                continue
            count = 0
            for index, node in enumerate(G.edges_from(i)):
                if node in infected:
                    count += 1
            if (count / (index + 1)) >= q:
                terminate = False
                infected.add(i)
    
    return list(infected)

def threshold_cascade(G, S, num_iter = 5, complete=True):
    ceil = 1
    floor = 0
    iter = 0
    while(iter < num_iter):
        q = (ceil + floor) / 2
        if len(contagion_brd(G, S, q)) == G.number_of_nodes():
            if complete:
                iter += 1
            floor = q
        else:
            if not complete:
                iter += 1
            ceil = q
    return q

def q_completecascade_graph_fig4_1_left():
    '''Return a float q s.t. the left graph in Figure 4.1 cascades completely.'''
    G = UndirectedGraph(4)
    for i in range(3):
        G.add_edge(i, i +1)
    S  = [0, 1]
    q = threshold_cascade(G, S)
    return q

def q_incompletecascade_graph_fig4_1_left():
    '''Return a float q s.t. the left graph in Figure 4.1 does not cascade completely.'''
    G = UndirectedGraph(4)
    for i in range(3):
        G.add_edge(i, i +1)
    S  = [0, 1]
    q = threshold_cascade(G, S, complete=False)
    return q

def q_completecascade_graph_fig4_1_right():
    '''Return a float q s.t. the right graph in Figure 4.1 cascades completely.'''
    G = UndirectedGraph(7)
    for i in range(3):
        G.add_edge(i, i +1)

    for i in range(1, 4):
        G.add_edge(i, i + 3)

    S = [0, 1, 4]
    q = threshold_cascade(G, S)
    return q

def q_incompletecascade_graph_fig4_1_right():
    '''Return a float q s.t. the right graph in Figure 4.1 does not cascade completely.'''
    G = UndirectedGraph(7)
    for i in range(3):
        G.add_edge(i, i +1)

    for i in range(1, 4):
        G.add_edge(i, i + 3)
    
    S = [0, 1, 4]
    q = threshold_cascade(G, S, complete=False)
    return q

def main():
    q = q_completecascade_graph_fig4_1_left()
    print(f"With adoption threshold {q} complete cascade has occurred")
    q = q_incompletecascade_graph_fig4_1_left()
    print(f"With adoption threshold {q} complete cascade has not occurred")

    q = q_completecascade_graph_fig4_1_right()
    print(f"With adoption threshold {q} complete cascade has occurred")
    q = q_incompletecascade_graph_fig4_1_right()
    print(f"With adoption threshold {q} complete cascade has not occurred")

    G = create_fb_graph()
    # === Problem 8(b) === #
    avg_num_infected = 0
    q = 0.1
    for _ in range(100):
        S = np.random.randint(G.number_of_nodes(), size=10)
        num_infected = len(contagion_brd(G, S, q))
        avg_num_infected += num_infected
    avg_num_infected /= 100
    print(f"Average number of infected nodes is {avg_num_infected}")
    # === Problem 8(c) === #
    for k in range(0, 250, 10):
        num_infected_li = []
        for q in np.arange(0, 0.5, 0.05):
            avg_num_infected = 0
            for _ in range(100):
                S = np.random.randint(G.number_of_nodes(), size=k)
                num_infected = len(contagion_brd(G, S, q))
                avg_num_infected += num_infected
            avg_num_infected /= 100
            num_infected_li.append(avg_num_infected)
        plt.plot(np.arange(0, 0.5, 0.05), num_infected_li, label = f"|S| = {k}")

    plt.xlabel("Adoption Threshold")
    plt.ylabel("Average Number of Nodes Infected")
    plt.title("Average Number Infected Against Adoption Threshold")
    plt.legend(loc="upper left")
    plt.show()

    # === OPTIONAL: Bonus Question 2 === #
    # TODO: Put analysis code here
    pass

# === OPTIONAL: Bonus Question 2 === #
def min_early_adopters(G, q):
    '''Given an undirected graph G, and float threshold q, approximate the 
       smallest number of early adopters that will call a complete cascade.
       Return an integer between [0, G.number_of_nodes()]'''
    pass

if __name__ == "__main__":
    main()
