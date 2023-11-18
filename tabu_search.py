#!/usr/bin/env python3

import os
import pandas as pd
import random as rd
import numpy as np
from itertools import combinations
import math
from argparse import ArgumentParser

class TabuSearch():
    def __init__(self, path, seed, tabu_tenure, debug_mode):
        self.debug_mode = debug_mode
        self.path = path
        self.seed = seed
        self.tabu_tenure = tabu_tenure
        self.instance_dict = self.input_data()
        self.initial_solution = self.get_initial_solution()
        self.tabu_str, self.best_solution, self.best_objvalue = self.tabu_search()


    def input_data(self):
        return pd.read_csv(self.path, names=['Job', 'weight', 'processing_time', "due_date"],
                                 index_col=0).to_dict('index')

    def get_tabu_structure(self):
        dict = {}
        for swap in combinations(self.instance_dict.keys(), 2):
            dict[swap] = {'tabu_time': 0, 'move_value': 0}
        return dict

    def get_initial_solution(self):
        n_jobs = len(self.instance_dict) # Number of jobs
        # Producing a random schedule of jobs
        initial_solution = list(range(1, n_jobs+1))
        rd.seed(self.seed)
        rd.shuffle(initial_solution)
        if self.debug_mode == True:
            print(f"Initial random solution: {initial_solution}")
        return initial_solution

    def cost_function(self, solution):
        dict = self.instance_dict
        t = 0   #starting time
        objfun_value = 0
        for job in solution:
            C_i = t + dict[job]["processing_time"]  # Completion time
            d_i = dict[job]["due_date"]   # due date of the job
            T_i = max(0, C_i - d_i)    #tardiness for the job
            W_i = dict[job]["weight"]  # job's weight

            objfun_value +=  W_i * T_i
            t = C_i
        if self.debug_mode == True:
            print("\n","#"*8, "The Objective function value for {} solution schedule is: {}".format(solution ,objfun_value),"#"*8)
        return objfun_value

    def swap_move(self, solution, i ,j):
        solution = solution.copy()
        # job index in the solution:
        i_index = solution.index(i)
        j_index = solution.index(j)
        #Swap
        solution[i_index], solution[j_index] = solution[j_index], solution[i_index]
        return solution

    def tabu_search(self):
        # Parameters:
        tenure =self.tabu_tenure
        tabu_structure = self.get_tabu_structure()  # Initialize the data structures
        best_solution = self.initial_solution
        best_objvalue = self.cost_function(best_solution)
        current_solution = self.initial_solution
        current_objvalue = self.cost_function(current_solution)
        if self.debug_mode == True:
            print(f"Short-term memory TS with Tabu Tenure: {tenure}")
            print(f"Initial Solution: {current_solution}")
            print(f"Initial Objvalue: {current_objvalue}")
        iter = 1
        terminate = 0
        while terminate < 100:
            if self.debug_mode == True:
                print(f"iter {iter}")
                print(f"Current_Objvalue: {current_objvalue}")
                print(f"Best_Objvalue: {best_objvalue}")
            # Searching the whole neighborhood of the current solution:
            for move in tabu_structure:
                candidate_solution = self.swap_move(current_solution, move[0], move[1])
                candidate_objvalue = self.cost_function(candidate_solution)
                tabu_structure[move]['move_value'] = candidate_objvalue

            # Admissible move
            while True:
                # select the move with the lowest ObjValue in the neighborhood (minimization)
                best_move = min(tabu_structure, key =lambda x: tabu_structure[x]['move_value'])
                move_value = tabu_structure[best_move]["move_value"]
                tabu_time = tabu_structure[best_move]["tabu_time"]
                # Not Tabu
                if tabu_time < iter:
                    # make the move
                    current_solution = self.swap_move(current_solution, best_move[0], best_move[1])
                    current_objvalue = self.cost_function(current_solution)
                    # Best Improving move
                    if move_value < best_objvalue:
                        best_solution = current_solution
                        best_objvalue = current_objvalue
                        terminate = 0
                    else:
                        if self.debug_mode == True:
                            print(f"Termination: {terminate} ## best_move: {best_move}, Objvalue: {current_objvalue} => Least non-improving => Admissible")
                        terminate += 1
                    # update tabu_time for the move
                    tabu_structure[best_move]['tabu_time'] = iter + tenure
                    iter += 1
                    break
                # If tabu
                else:
                    # Aspiration
                    if move_value < best_objvalue:
                        # make the move
                        current_solution = self.swap_move(current_solution, best_move[0], best_move[1])
                        current_objvalue = self.cost_function(current_solution)
                        best_solution = current_solution
                        best_objvalue = current_objvalue
                        if self.debug_mode == True:
                            print(f"best_move: {best_move}, Objvalue: {current_objvalue} => Aspiration => Admissible")
                        terminate = 0
                        iter += 1
                        break
                    else:
                        tabu_structure[best_move]["move_value"] = float('inf')
                        if self.debug_mode == True:
                            print(f"best_move: {best_move}, Objvalue: {current_objvalue} => Tabu => Inadmissible")
                        continue
        print(f"Performed iterations: {iter}")
        print(f"Best found Solution: {best_solution}")
        print(f"Cost value: {best_objvalue}")
        return tabu_structure, best_solution, best_objvalue

parser = ArgumentParser()
parser.add_argument("--data", help="path to file with data", default="data.csv")
parser.add_argument("--debug", help="Debug mode", default="no", choices=["yes", "no"])

args = parser.parse_args()

datafile = args.data

debug_mode = False
if args.debug == "yes": debug_mode = True

print(f"Executing for datafile: {datafile}")
test = TabuSearch(path=datafile, seed = 2012, tabu_tenure=3, debug_mode=False)