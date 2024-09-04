from typing import List
import numpy as np

class Node:
    def __init__(self):
        self.neighbors = []

    def add_neighbor(self, node):
            self.neighbors.append(node)

class UndirectedGraph:

    def __init__(self, adjency_matrix: List[List[bool]]):
        self.nodes = [Node() for _ in adjency_matrix]

        for row in range(len(adjency_matrix)):
             for col in range(len(adjency_matrix)):
                  if adjency_matrix[row][col]:
                       self.nodes[row].add_neighbor(self.nodes[col])

def create_graph(n, p):
    
    adjency_matrix = np.random.uniform(0, 1, size = (n, n)) < p
    
    return UndirectedGraph(adjency_matrix)



