import sys

import numpy
import numpy as np
import matplotlib.pyplot as plt


class VRPTabuSearch:
    def __init__(self, vehicle_count, tabu_tenure):
        self.distances = []
        self.solution = []
        self.node_count = 0
        self.vehicle_count = vehicle_count
        self.tabu_tenure = tabu_tenure

    def read_distance_graph(self, filename):
        with open(filename, 'rb') as f:
            self.distances = np.load(f)
        if self.node_count != 0 and self.node_count != self.distances.shape[0]:
            sys.exit('Different size of distance graph and nodes list')
        self.node_count = self.distances.shape[0]

    def read_node_list(self, filename):
        print('not implemented')

    def get_candidate_list(self, start_point, percentage, is_visited):
        candidate_list_size = min(int(self.node_count * percentage / 100) + 1, self.node_count-1)
        candidate_list = np.argpartition(self.distances[start_point], candidate_list_size)[:candidate_list_size]
        for candidate in candidate_list:
            if is_visited[candidate]:
                candidate_list = np.delete(candidate_list, candidate)
        return candidate_list

    def get_closest_point(self, start_point, is_visited):
        sorted_args = np.argsort(self.distances[start_point])
        for i in range(len(is_visited)):
            if not is_visited[sorted_args[i]]:
                return sorted_args[i]

    def initial_solution(self):
        routes = np.zeros((self.vehicle_count, self.node_count+1), dtype=int)
        is_visited = np.full(self.node_count, False)
        is_visited[0] = True
        all_is_visited = False
        i = 0
        while not all_is_visited:
            for vehicle in range(self.vehicle_count):
                start_point = routes[vehicle][i]
                next_point = self.get_closest_point(start_point, is_visited)
                routes[vehicle][i+1] = next_point
                is_visited[next_point] = True
                if all(is_visited):
                    all_is_visited = True
                    break
            i += 1
        return routes

    def cost_function(self, routes):
        cost = 0
        for i in range(self.vehicle_count):
            j = 0
            while routes[i][j+1] != 0:
                cost += self.distances[routes[i][j]][routes[i][j+1]]
                j += 1
            cost += self.distances[0][routes[i][j]]
        return cost

    def two_swap(self, routes, node_i, node_j):
        i_index = numpy.where(routes == node_i)
        j_index = numpy.where(routes == node_j)
        routes[i_index[0][0]][i_index[1][0]], routes[j_index[0][0]][j_index[1][0]] = (
            routes[j_index[0][0]][j_index[1][0]], routes[i_index[0][0]][i_index[1][0]])
        return routes


    def two_insert(self, routes, node_i, node_j):
        # put node_j before node_i
        i_index = numpy.where(routes == node_i)
        j_index = numpy.where(routes == node_j)
        routes[i_index[0][0]] = np.concatenate((routes[i_index[0][0]][:i_index[1][0]], [routes[j_index[0][0]][j_index[1][0]]], routes[i_index[0][0]][i_index[1][0]:-1]))
        routes[j_index[0][0]] = np.concatenate((routes[j_index[0][0]][:j_index[1][0]], routes[j_index[0][0]][j_index[1][0]+1:], [0]))
        return routes


vrp_tabu = VRPTabuSearch(2, 5)
vrp_tabu.read_distance_graph('data_graph.npy')
solution = vrp_tabu.initial_solution()
print(solution)
cost = vrp_tabu.cost_function(solution)
print(cost)
solution = vrp_tabu.two_insert(solution, 2, 1)
print(solution)
cost = vrp_tabu.cost_function(solution)
print(cost)
