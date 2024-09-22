from typing import List
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
from tqdm  import tqdm

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

     for _ in tqdm(range(num_samples)):
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

def create_graph_simple(N):
     adjency_matrix = np.zeros((N, N))
     for i in range(N):
          if i + 1 < N:
               adjency_matrix[i][i + 1] = 1
               adjency_matrix[i + 1][i] = 1
          if i - 1 < 0:
               adjency_matrix[i][i - 1] = 1
               adjency_matrix[i-1][i] = 1
     
     print(adjency_matrix)
     return UndirectedGraph(adjency_matrix)  

def create_graph_v2(X, Y):
     adjency_matrix = np.zeros((X + Y, X + Y))
     for i in range(X+Y):
          if i + 1 < X+Y:
               adjency_matrix[i][i + 1] = 1
               adjency_matrix[i + 1][i] = 1
          if i - 1 > 0:
               adjency_matrix[i][i - 1] = 1
               adjency_matrix[i-1][i] = 1

     ones = np.ones((X, X))

     adjency_matrix[:X,:X] = ones
     
     print(adjency_matrix)
     return UndirectedGraph(adjency_matrix)           

#X = c
#Y=2*c
#c =10 
c=20
x=4*c**3
y = 2*c
G = create_graph_v2(x,y)

print(y/avg_shortest_path(G, num_samples=50000))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
"""
import matplotlib.pyplot as plt


avg_shortest_paths = []

for i in range(5, 20):
     G = create_graph_simple(i)
     avg_shortest_paths.append(avg_shortest_path(G, num_samples=10000))

plt.plot(list(range(5, 20)), avg_shortest_paths)
plt.show()
#print(avg_shortest_path(G, num_samples=10000)) 

"""