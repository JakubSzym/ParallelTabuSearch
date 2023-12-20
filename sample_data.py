#!/usr/bin/env python3

import random
from argparse import ArgumentParser
import numpy as np
import math

filename_graph = 'data_graph.npy'
filename = 'nodes.txt'
nodes_count = 100
minX = 0
maxX = 50
minY = 0
maxY = 50


with open(filename, "w") as file:
  file.write("label,x,y\n")
  x_depot = 0
  y_depot = 0
  nodes = [(x_depot, y_depot)]
  file.write(f"Depot,{x_depot},{y_depot}\n")
  for i in range(nodes_count-1):
    x = random.uniform(minX, maxX)
    y = random.uniform(minY, maxY)
    if not (x, y) in nodes:
        nodes.append((x, y))
        i += 1
    file.write(f"N{i+1},{x},{y}\n")

data_matrix = np.zeros((nodes_count, nodes_count))

for i in range(nodes_count):
    for j in range(nodes_count):
        data_matrix[i][j] = data_matrix[j][i] = math.dist(nodes[i], nodes[j])

with open(filename_graph, 'wb+') as f:
    np.save(f, data_matrix)
