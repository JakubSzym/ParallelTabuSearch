import sys
import numpy
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time


class VRPTabuSearch:
    def __init__(self, vehicle_count, tabu_tenure, max_iterations, candidate_list_percentage, distance_graph_filename):
        self.distances = []
        self.solution = []
        self.node_count = 0
        self.vehicle_count = vehicle_count
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.candidate_list_percentage = candidate_list_percentage
        self.candidate_list_size = 0
        self.read_distance_graph(distance_graph_filename)
        self.current_solution = self.initial_solution()
        self.tabu_list = np.zeros((self.tabu_tenure, self.current_solution.shape[0], self.current_solution.shape[1]), dtype=int)

    def read_distance_graph(self, filename):
        with open(filename, 'rb') as f:
            self.distances = np.load(f)
        if self.node_count != 0 and self.node_count != self.distances.shape[0]:
            sys.exit('Different size of distance graph and nodes list')
        self.node_count = self.distances.shape[0]
        self.candidate_list_size = min(int(self.node_count * self.candidate_list_percentage / 100) + 1, self.node_count - 1)

    def read_node_list(self, filename):
        print('not implemented')

    def get_candidate_list(self, start_point):
        candidate_list = np.argpartition(self.distances[start_point], self.candidate_list_size)[:self.candidate_list_size]
        candidate_list = np.delete(candidate_list, np.where(candidate_list == start_point))
        candidate_list = np.delete(candidate_list, np.where(candidate_list == 0))
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
        routes[i_index[0][0]][i_index[1][0]], routes[j_index[0][0]][j_index[1][0]] = routes[j_index[0][0]][j_index[1][0]], routes[i_index[0][0]][i_index[1][0]]
        return routes

    def two_insert(self, routes, node_i, node_j):
        # put node_j before node_i
        i_index = numpy.where(routes == node_i)
        j_index = numpy.where(routes == node_j)
        if i_index[0][0] == j_index[0][0]:
            if i_index[1][0] < j_index[1][0]:
                routes[i_index[0][0]] = np.concatenate((routes[i_index[0][0]][:i_index[1][0]],
                                                       [routes[i_index[0][0]][j_index[1][0]]],
                                                       routes[i_index[0][0]][i_index[1][0]:j_index[1][0]],
                                                       routes[i_index[0][0]][j_index[1][0]+1:]))
            if i_index[1][0] > j_index[1][0]:
                routes[i_index[0][0]] = np.concatenate((routes[i_index[0][0]][:j_index[1][0]],
                                                       routes[i_index[0][0]][j_index[1][0]+1:i_index[1][0]],
                                                       [routes[i_index[0][0]][j_index[1][0]]],
                                                       routes[i_index[0][0]][i_index[1][0]:]))
            return routes

        routes[i_index[0][0]] = np.concatenate((routes[i_index[0][0]][:i_index[1][0]], [routes[j_index[0][0]][j_index[1][0]]], routes[i_index[0][0]][i_index[1][0]:-1]))
        routes[j_index[0][0]] = np.concatenate((routes[j_index[0][0]][:j_index[1][0]], routes[j_index[0][0]][j_index[1][0]+1:], [0]))
        return routes

    def is_tabu(self, solution, tabu_list):
        for tabu in tabu_list:
            if (solution == tabu).all():
                return True
        return False

    def explore_nodes_neighbors(self, node):
        best_neighbor_solution = []
        best_neighbor_cost = sys.maxsize
        candidate_list = self.get_candidate_list(node)
        for candidate in candidate_list:
            # explore swap with candidate
            temp_solution = self.two_swap(np.copy(self.current_solution), node, candidate)
            temp_cost = self.cost_function(temp_solution)
            if temp_cost < best_neighbor_cost:
                if not self.is_tabu(temp_solution, self.tabu_list):
                    best_neighbor_cost = temp_cost
                    best_neighbor_solution = np.copy(temp_solution)
            # explore insertion of candidate
            temp_solution = self.two_insert(np.copy(self.current_solution), node, candidate)
            temp_cost = self.cost_function(temp_solution)
            if temp_cost < best_neighbor_cost:
                if not self.is_tabu(temp_solution, self.tabu_list):
                    best_neighbor_cost = temp_cost
                    best_neighbor_solution = np.copy(temp_solution)
        return best_neighbor_cost, best_neighbor_solution

    def tabu_search_parallel(self):
        best_solution = numpy.copy(self.current_solution)
        self.tabu_list[0] = best_solution
        best_cost = self.cost_function(best_solution)
        iteration = 0
        max_iteration_stop = False
        while not max_iteration_stop:
            with Pool(5) as pool:
                results = pool.map(self.explore_nodes_neighbors, range(1, self.node_count))
            costs, solutions = zip(*list(results))
            best_result_id = min(range(len(costs)), key=costs.__getitem__)
            best_neighbor_cost = costs[best_result_id]
            best_neighbor_solution = np.asarray(solutions[best_result_id])
            self.tabu_list = np.concatenate(([best_neighbor_solution], self.tabu_list[:-1]))
            self.current_solution = np.copy(best_neighbor_solution)
            if best_neighbor_cost < best_cost:
                best_solution = np.copy(best_neighbor_solution)
                best_cost = np.copy(best_neighbor_cost)

            if iteration < self.max_iterations:
                iteration += 1
            else:
                max_iteration_stop = True

        return best_solution, best_cost

    def tabu_search(self):
        best_solution = numpy.copy(self.current_solution)
        self.tabu_list[0] = best_solution
        best_cost = self.cost_function(best_solution)
        iteration = 0
        max_iteration_stop = False
        while not max_iteration_stop:
            best_neighbor_solution = []
            best_neighbor_cost = best_cost * 10
            for node_iter in range(1, self.node_count):
                candidate_list = self.get_candidate_list(node_iter)
                for candidate in candidate_list:
                    # explore swap with candidate
                    temp_solution = self.two_swap(np.copy(self.current_solution), node_iter, candidate)
                    temp_cost = self.cost_function(temp_solution)
                    if temp_cost < best_neighbor_cost:
                        if not self.is_tabu(temp_solution, self.tabu_list):
                            best_neighbor_cost = temp_cost
                            best_neighbor_solution = np.copy(temp_solution)
                    # explore insertion of candidate
                    temp_solution = self.two_insert(np.copy(self.current_solution), node_iter, candidate)
                    temp_cost = self.cost_function(temp_solution)
                    if temp_cost < best_neighbor_cost:
                        if not self.is_tabu(temp_solution, self.tabu_list):
                            best_neighbor_cost = temp_cost
                            best_neighbor_solution = np.copy(temp_solution)

            self.tabu_list = np.concatenate(([best_neighbor_solution], self.tabu_list[:-1]))
            self.current_solution = np.copy(best_neighbor_solution)
            if best_neighbor_cost < best_cost:
                best_solution = np.copy(best_neighbor_solution)
                best_cost = np.copy(best_neighbor_cost)

            if iteration < self.max_iterations:
                iteration += 1
            else:
                max_iteration_stop = True

        return best_solution, best_cost


if __name__ == '__main__':
    vrp_tabu = VRPTabuSearch(10, 5, 200, 50, 'data_graph.npy')
    solution = vrp_tabu.initial_solution()
    print(solution)
    cost = vrp_tabu.cost_function(solution)
    print(cost)
    start_time = time.time()
    best_solution, best_cost = vrp_tabu.tabu_search_parallel()
    print(str((time.time() - start_time)) + ' seconds in parallel')
    start_time = time.time()
    best_solution, best_cost = vrp_tabu.tabu_search()
    print(str((time.time() - start_time)) + ' seconds in one thread')
    print(best_cost)
    print(best_solution)



