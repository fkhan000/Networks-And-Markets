from typing import List
import numpy as np
from collections import deque


class UndirectedGraph:

    def __init__(self, adjency_matrix: List[List[bool]]):
        
        self.nodes = [[] for _ in range(len(adjency_matrix))]
        
        for row in range(len(adjency_matrix)):
             for col in range(len(adjency_matrix)):
                  if adjency_matrix[row][col]:
                       self.nodes[row].append(col)

def create_graph(n, p):
     adjency_matrix = np.random.rand(n, n) < p
     
     return UndirectedGraph(adjency_matrix)

def shortest_path(G: UndirectedGraph, i: int, j: int) -> int:

     queues = [deque([(i, 0)]), deque([(j, 0)])]
     visited = [{i:0}, {j:0}]

     while queues[0] and queues[1]:

          for index in range(2):
               current_node, path_length = queues[index].popleft()

               for neighbor in G.nodes[current_node]:
                    if neighbor not in visited[index]:
                         visited[index][neighbor] = path_length + 1
                         queues[index].append((neighbor, path_length + 1))
                    if neighbor in visited[(index + 1) % 2]:
                         return visited[index][neighbor] + visited[(index + 1) % 2][neighbor]
                    
     return -1

def avg_shortest_path(G: UndirectedGraph, num_samples=1000):

     indices = list(range(len(G.nodes)))
     avg_length = 0

     for _ in range(num_samples):
          length = -1
          while(length == -1):
               i, j = np.random.choice(indices, 2, replace=False)
               length = shortest_path(G, i, j)
          avg_length += length
     return avg_length / num_samples

G = create_graph(1000, 0.8)
import time

start = time.time()
print(avg_shortest_path(G))
print(time.time()-start)