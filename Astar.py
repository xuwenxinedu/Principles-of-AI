import numpy as np
import csv


class AStarPathPlanner(object):
    def __init__(self, nodes_path, edges_path):
        # initialize heuristic_cost_to_go for each node
        self.heuristic_cost_to_go = []
        with open(nodes_path, "rt") as f_obj:
            contents = csv.reader(f_obj)
            for row in contents:
                # ignore comments in file
                if(row[0][0] != '#'):
                    self.heuristic_cost_to_go.append(float(row[3]))
        self.nodes_number = len(self.heuristic_cost_to_go)

        # 给出下面程序的功能
        # 读edges表，得到相邻点并得到相邻两点间的直接距离
        # 
        self.neighbour_table = {}
        self.cost = np.zeros((self.nodes_number, self.nodes_number))
        with open(edges_path, "rt") as f_obj:
            contents = csv.reader(f_obj)
            for row in contents:
                # ignore comments in file
                if(row[0][0] != '#'):
                    node1 = int(row[0])
                    node2 = int(row[1])
                    # construct neighbour information
                    if node1 not in self.neighbour_table:
                        self.neighbour_table[node1] = []
                    self.neighbour_table[node1].append(node2)
                    if node2 not in self.neighbour_table:
                        self.neighbour_table[node2] = []
                    self.neighbour_table[node2].append(node1)

                    # construct neighbour cost information
                    cost = float(row[2])
                    self.cost[node1-1][node2-1] = cost
                    self.cost[node2-1][node1-1] = cost
       
    def search_for_path(self):
        self.OPEN = [1]
        self.CLOSED = []
        self.est_total_cost = [float("inf")] * (self.nodes_number)
        self.past_cost = [0] * (self.nodes_number)
        self.past_cost[0] = 0
        self.path = []
        # store node parent information
        self.parent = {}
        # set to infinity for node 2...N
        for i in range(1, self.nodes_number):
            self.past_cost[i] = float("inf")
        while self.OPEN:
            current = self.OPEN.pop(0)
            self.CLOSED.append(current)
            # path has been found
            if current == self.nodes_number:
                # reconstruct path by parent node
                while True:
                    self.path.append(current)
                    if current == 1:
                        break
                    current = self.parent[current]
                return True, self.path

            for nbr in self.neighbour_table[current]:
                if nbr not in self.CLOSED:
                #完成扩展节点current，并根据评价函数重新排序open表的过程    
                    if nbr not in self.OPEN: 
                        self.OPEN.append(nbr)
                        self.parent[nbr] = current
                        self.past_cost[nbr-1] = self.past_cost[current-1] + self.cost[current-1][nbr-1]
                self.OPEN.sort(key=lambda x:(self.past_cost[x-1]+self.heuristic_cost_to_go[x-1]))    
        return False, []

    def save_path_to_file(self, path):
        with open(path, 'wt') as f_obj:
            writer = csv.writer(f_obj, delimiter=',')
            writer.writerow(self.path[::-1])


if __name__ == "__main__":
    planner = AStarPathPlanner("nodes.csv", "edges.csv")
    success, path = planner.search_for_path()
    if success:
        print (path[::-1])
        planner.save_path_to_file("path.csv")
    else:
        print ("no solution found")