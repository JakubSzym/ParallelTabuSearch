import sys
import random
import numpy as np


class VRPTabuSearch:
    def __init__(self, vehicle_count):
        self.distances = []
        self.solution = []
        self.node_count = 0
        self.vehicle_count = vehicle_count

    def read_distance_graph(self, filename):
        with open(filename, 'rb') as f:
            self.distances = np.load(f)
        if self.node_count != 0 and self.node_count != self.distances.shape[0]:
            sys.exit('Different size of distance graph and nodes list')
        self.node_count = self.distances.shape[0]

    def read_node_list(self, filename):
        print('not implemented')

    def get_candidate_list(self, start_point, percentage, is_visited):
        candidate_list_size = min(int(self.node_count * percentage / 100) + 1, self.node_count)
        candidate_list = np.argpartition(self.distances[start_point], candidate_list_size)
        candidate_list = np.delete(candidate_list, start_point)
        for candidate in candidate_list:
            if is_visited[candidate]:
                candidate_list = np.delete(candidate_list, candidate)
        return candidate_list

    def initial_solution(self):
        routes = np.zeros((self.vehicle_count, self.node_count))
        is_visited = np.full(self.node_count, False)
        is_visited[0] = True
        all_is_visited = False
        i = 0
        while not all_is_visited:
            for vehicle in range(self.vehicle_count):
                start_point = routes[vehicle][i]
                candidate_list = self.get_candidate_list(start_point, 100, is_visited)
                next_point = candidate_list[0]
                routes[vehicle][i+1] = next_point
                is_visited[next_point] = True
                if all(is_visited):
                    all_is_visited = True
                    break
            i += 1
        return routes





