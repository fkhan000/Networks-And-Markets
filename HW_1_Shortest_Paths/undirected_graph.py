from typing import List
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt


class UndirectedGraph:

    def __init__(self, adjency_matrix: List[List[bool]]):
        
        self.nodes = [set() for _ in range(len(adjency_matrix))]
        
        for row in range(len(adjency_matrix)):
             for col in range(len(adjency_matrix)):
                  if adjency_matrix[row][col]:
                       self.nodes[row].add(col)

def create_graph(n, p):
     adjency_matrix = np.random.rand(n, n) < p
     
     return UndirectedGraph(adjency_matrix)

def shortest_path(G: UndirectedGraph, i: int, j: int) -> int:

     queues = [deque([(i, 0)]), deque([(j, 0)])]
     visited = [{i:0}, {j:0}]

     while queues[0] and queues[1]:
          
          for index in range(2):
               current_node, path_length = queues[index].popleft()

               if current_node in visited[(index + 1) % 2]:
                    return path_length + visited[(index + 1) % 2][current_node]

               for neighbor in G.nodes[current_node]:
                    if neighbor not in visited[index]:
                         visited[index][neighbor] = path_length + 1
                         queues[index].append((neighbor, path_length + 1))                    
     return -1

def probability_connected_edge(G: UndirectedGraph, num_samples=1000):
     indices = list(range(len(G.nodes)))

     prob_connected = 0

     for _ in range(num_samples):
          i, j = np.random.choice(indices, 2, replace=False)
          if j in G.nodes[i]:
               prob_connected += 1
     return prob_connected / num_samples

def avg_shortest_path(G: UndirectedGraph, num_samples=1000):

     indices = list(range(len(G.nodes)))
     avg_length = 0

     for _ in range(num_samples):
          i, j = np.random.choice(indices, 2, replace=False)
          length = shortest_path(G, i, j)

          if length == -1:
               return -1
          avg_length += length
     return avg_length / num_samples

def simulate_varying_prob(prob_vals):

     avg_distances = []
     for p in prob_vals:
          avg_distance = -1
          while(avg_distance == -1):
               G = create_graph(1000, p)
               avg_distance = avg_shortest_path(G)          
          avg_distances.append(avg_distance)

     plt.plot(prob_vals, avg_distances)
     plt.show()

G = create_graph(1000, 0.1)
print(avg_shortest_path(G))