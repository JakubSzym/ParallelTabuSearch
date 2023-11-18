#!/usr/bin/env python3

import random
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--filename", "-f", help="output file")
parser.add_argument("--nodes",    help="number of nodes")
parser.add_argument("--minX",     help="minimal value of x coordinate")
parser.add_argument("--maxX",     help="maximal value of x coordinate")
parser.add_argument("--minY",     help="minimal value of y coordinate")
parser.add_argument("--maxY",     help="maximal value of y coordinate")

args = parser.parse_args()

filename = args.filename
nodes    = int(args.nodes)
minX     = int(args.minX)
maxX     = int(args.maxX)
minY     = int(args.minY)
maxY     = int(args.maxY)

with open(filename, "w") as file:
  file.write("label,x,y\n")
  x_depot = random.uniform(minX, maxX)
  y_depot = random.uniform(minY, maxY)
  file.write(f"Depot,{x_depot},{y_depot}\n")
  for i in range(nodes-1):
    x = random.uniform(minX, maxX)
    y = random.uniform(minY, maxY)
    file.write(f"N{i+1},{x},{y}\n")