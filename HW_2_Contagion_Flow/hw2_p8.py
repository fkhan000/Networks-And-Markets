
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
    
    # in order to optimize the contagion at each iteration we only look to see if the nodes
    # at the frontier of the contagion want to swtich

    infected = set(S)
    neighbor_set = set()

    # get neighbors of infected nodes
    for node in infected:
        for neighbor in G.nodes[node]:
            if neighbor not in infected:
                neighbor_set.add(neighbor)
    
    terminate = False
    while(not terminate):
        terminate = True
        neighbors_to_remove = set()
        neighbors_to_add = set()
        # for each neighbor to an infected node
        for node in neighbor_set:
            # we get its neighbors
            neighbors = G.nodes[node]
            # and find how many of them are infected
            num_infected = len(neighbors.intersection(infected))

            # if it exceeds the threshold
            if (num_infected / (len(neighbors))) >= q:
                # we turn off the terminate flag
                terminate = False
                # add it to our infected set
                infected.add(node)
                # and remove it from the frontier
                neighbors_to_remove.add(node)

                # and add its uninfected neighbors to the frontier
                neighbors_to_add.update(neighbors.difference(infected))
        # finally we update the neighbor set
        neighbor_set.difference_update(neighbors_to_remove)
        neighbor_set.update(neighbors_to_add)
    
    # once BRD terminates we return the list of infected nodes
    return list(infected)

def threshold_cascade(G, S, num_iter = 5, complete=True):
    # for a given q if we apply BRD and all of the nodes are infected
    # we are either at or below the requried adoption threshold. And if we apply
    # BRD and all nodes aren't infected then we are above the required adoption threshold.

    # So we can use this to perform binary search and very quickly get a good approximation 
    # of the minimum required adoption threshold

    ceil = 1
    floor = 0
    iter = 0
    while(iter < num_iter):
        q = (ceil + floor) / 2
        # if all nodes have been infected
        if len(contagion_brd(G, S, q)) == G.number_of_nodes():
            if complete:
                # we update the iter count here 
                # so that we terminate at a value of q that results in all nodes being infected
                iter += 1
            # we need to increase q
            floor = q
        else:
            # else we decrease q
            if not complete:
                # and update iter count if we want a value of q that 
                # results in not all nodes being infected
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
    # over several trials
    for _ in range(100):
        # we select a random set of adopters and perform BRD with q = 0.1
        S = np.random.randint(G.number_of_nodes(), size=10)
        num_infected = len(contagion_brd(G, S, q))
        # and obtain the average number of nodes infected
        avg_num_infected += num_infected
    avg_num_infected /= 100
    print(f"Average number of infected nodes is {avg_num_infected}")
    # === Problem 8(c) === #
    # for varying number of adopters
    for k in range(0, 250, 10):
        num_infected_li = []
        # and varying values of q
        for q in np.arange(0, 0.5, 0.05):
            avg_num_infected = 0
            # we calculate the average number of infected
            for _ in range(100):
                S = np.random.randint(G.number_of_nodes(), size=k)
                num_infected = len(contagion_brd(G, S, q))
                avg_num_infected += num_infected
            avg_num_infected /= 100
            num_infected_li.append(avg_num_infected)
        # and then plot the average number of nodes infected against the adoption threshold
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
