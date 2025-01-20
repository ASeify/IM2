# version: 2.0.0
import networkx as nx
class K_Shell_Calculate:
    def __init__(self, network_graph:nx.Graph):
        self.network_graph = network_graph

    def get_k_shell_info(self):
        temp_graph = self.network_graph.copy()
        nx.set_node_attributes(temp_graph, 0, "k_shell")
        nx.set_node_attributes(temp_graph, 0, "k_shell_itr")
        nx.set_node_attributes(temp_graph, 1, "vote_power")
        repeat_flag = True
        k = 0  # K-Shall number
        k_shell_list = [[0]]
        while len(list(self.network_graph.nodes())) > 0:
            nk = 1  # K-Shall iteration number
            while repeat_flag:
                repeat_flag = False
                nodes_list = dict([degree for (degree) in self.network_graph.degree()])
                for current_node, current_node_degree in nodes_list.items():
                    if current_node_degree <= k:
                        temp_graph.nodes[current_node]["k_shell"] = k
                        temp_graph.nodes[current_node]["k_shell_itr"] = nk
                        neighbors = list(self.network_graph.neighbors(current_node))
                        uper_level_neighbors = []
                        for item in neighbors:
                            if nodes_list[item] > k:
                                uper_level_neighbors.append(item)
                        neighbors.clear()
                        neighbors_count = len(uper_level_neighbors)
                        for item in uper_level_neighbors:
                            temp_graph.nodes[item]["vote_power"] += (temp_graph.nodes[current_node]["vote_power"]/neighbors_count)
                        self.network_graph.remove_node(current_node)
                        repeat_flag = True
                        if len(k_shell_list) == k:
                            k_shell_list.append([])
                            k_shell_list[k].append(1)
                        elif len(k_shell_list) > k:
                            if len(k_shell_list[k]) == (nk - 1):
                                k_shell_list[k].append(1)
                            elif len(k_shell_list[k]) > (nk - 1):
                                k_shell_list[k][nk - 1] += 1
                nk += 1
            repeat_flag = True
            if len(k_shell_list) == k:
                k_shell_list.append([])
                k_shell_list[k].append(0)
            k += 1

        temp_graph.graph["k_shell_info"] = k_shell_list

        return temp_graph

    def get_subgraph_k_shell_info(self, subgraph:nx.Graph.subgraph):
        temp_graph = subgraph.copy()
        nx.set_node_attributes(temp_graph, 0, "k_shell")
        nx.set_node_attributes(temp_graph, 0, "k_shell_itr")
        nx.set_node_attributes(temp_graph, 1, "vote_power")
        repeat_flag = True
        k = 0  # K-Shall number

        while len(list(subgraph.nodes())) > 0:
                nk = 1  # K-Shall iteration number
                while repeat_flag:
                    repeat_flag = False
                    nodes_list = dict([degree for (degree) in subgraph.degree()])
                    for current_node, current_node_degree in nodes_list.items():
                        if current_node_degree <= k:
                            temp_graph.nodes[current_node]["k_shell"] = k
                            temp_graph.nodes[current_node]["k_shell_itr"] = nk
                            neighbors = list(subgraph.neighbors(current_node))
                            uper_level_neighbors = []
                            for item in neighbors:
                                if nodes_list[item] > k:
                                    uper_level_neighbors.append(item)
                            neighbors.clear()
                            # neighbors_count = len(uper_level_neighbors)
                            for item in uper_level_neighbors:
                                temp_graph.nodes[item]["vote_power"] += 1
                            subgraph.remove_node(current_node)
                            repeat_flag = True
                            
                    nk += 1
                repeat_flag = True
                k += 1

        return temp_graph