# version: 3.5.0

import pandas as pd
import networkx as nx
import math
from tqdm import tqdm

class Network_Node_Centrality:

    @staticmethod
    def cluster_coefficient(graphs_of_network:list[nx.Graph], nodes:list[int | str] | None)->None:
            if nodes is None:
                for graph in graphs_of_network:
                    nx.set_node_attributes(graph, nx.clustering(graph), 'clustering')
            else:
                for node in nodes:
                    for graph in graphs_of_network:
                       if node in graph and graph.nodes[node]["clustering"] is None:
                            graph.nodes[node]["clustering"] = nx.clustering(graph, node)
            pass
    
    @staticmethod
    def get_top_neighbors_of_neighbors_union(graph:nx.Graph, node:list[int | str])->list:
        neighbors_union = []
        node_neighbors = list(graph.neighbors(node))
        neighbors_degree = graph.degree(node_neighbors)
        neighbors_degree = sorted(neighbors_degree,key=lambda x: x[1], reverse=True)
        top_degree_neghbors = [neighbors_degree[0]]
        for item in neighbors_degree:
            if top_degree_neghbors[0][0] != item[0]:
                if top_degree_neghbors[0][1] == item[1]:
                    top_degree_neghbors.append(item)
                else:
                    neighbors_degree.clear()
                    break
        
        for item in top_degree_neghbors:
            neighbors_union += list(graph.neighbors(item[0]))
        
        neighbors_union = set(neighbors_union)
        return neighbors_union, len(top_degree_neghbors)

    @staticmethod
    def get_mean_neighbors_degree(graph:nx.Graph, node:list[int | str])->list:
        mean_neighbors_degree = 0
        node_neighbors = list(graph.neighbors(node))
        neighbors_degree = graph.degree(node_neighbors)
        for _, v in neighbors_degree:
            mean_neighbors_degree += v
        mean_neighbors_degree /= len(node_neighbors)
        return mean_neighbors_degree
    
    @staticmethod
    def get_sum_of_ndoes_degree(graph:nx.Graph, nodes:list[int | str])->list:
        degree_sum = 0
        degree_list = graph.degree(nodes)
        for _, v in degree_list:
            degree_sum += v
        return degree_sum

    @staticmethod
    def get_neighbors_degree_cc(graph:nx.Graph, node: int | str)->list:
        neighbors_score_sum = 0
        node_neighbors = list(graph.neighbors(node))
        neighbors_degree = graph.degree(node_neighbors)
        node_neighbors.clear()
        for k, v in neighbors_degree:
            neighbor_cc = graph.nodes[k]["clustering"]
            if neighbor_cc is None:                
                graph.nodes[k]["clustering"] = nx.clustering(graph, node)
                neighbor_cc = graph.nodes[k]["clustering"]
            neighbors_score_sum += (v * ((2 / (1 + neighbor_cc))/2))
        return neighbors_score_sum

    @staticmethod
    def node_nip(graphs_of_network:list[nx.Graph], nodes: list[int | str] = None) -> None:
        if nodes is None:
            for j, graph in enumerate(graphs_of_network):
                print(bcolors.HEADER + f'Layer {j}: {graph}' + bcolors.ENDC)
                with tqdm(list(graph.nodes), unit=" Node") as t_nodes:
                    for node in t_nodes:
                        if graph.nodes[node]["nip"] is None:
                            node_degree = graph.degree(node)
                            neighbors_degree_cc = Network_Node_Centrality.get_neighbors_degree_cc(graph, node)
                            node_cc = graph.nodes[node]["clustering"] # clustering coefficient
                            graph.nodes[node]["nip"] = node_degree + (neighbors_degree_cc * (2 / (1 + node_cc)))
        else:
                with tqdm(nodes, unit=" Node") as t_nodes:
                    for node in t_nodes:
                        for graph in graphs_of_network:
                            if node in graph and graph.nodes[node]["nip"] is None:
                                node_degree = graph.degree(node)
                                neighbors_degree_cc = Network_Node_Centrality.get_neighbors_degree_cc(graph, node)
                                node_cc = graph.nodes[node]["clustering"] # clustering coefficient
                                graph.nodes[node]["nip"] = node_degree + (neighbors_degree_cc * (2 / (1 + node_cc)))
        pass
    
    @staticmethod
    def sombor_index(graphs_of_network: list[nx.Graph], nodes: list[int | str] = None):
        if nodes is None:
            for j, graph in enumerate(graphs_of_network):
                print(bcolors.HEADER + f'Layer {j}: {graph}' + bcolors.ENDC)
                with tqdm(list(graph.nodes), unit=" Node") as t_nodes:
                    for node in t_nodes:
                        if graph.nodes[node]["sombor_index"] is None:
                            neighbors_list = list(graph.neighbors(node))
                            curent_node_degree = graph.degree(node)
                            sombor_index_score = 0
                            for neighbor in neighbors_list:
                                sombor_index_score += math.sqrt(
                                    (curent_node_degree ** 2) + (graph.degree(neighbor) ** 2))
                            graph.nodes[node]['sombor_index'] = sombor_index_score
        else:
            with tqdm(nodes, unit=" Node") as t_nodes:
                    for node in t_nodes:
                        for graph in graphs_of_network:
                            if node in graph and graph.nodes[node]["sombor_index"] is None:
                                neighbors_list = list(graph.neighbors(node))
                                curent_node_degree = graph.degree(node)
                                sombor_index_score = 0
                                for neighbor in neighbors_list:
                                    sombor_index_score += math.sqrt(
                                        (curent_node_degree ** 2) + (graph.degree(neighbor) ** 2))
                                graph.nodes[node]['sombor_index'] = sombor_index_score
        pass
    
    @staticmethod
    def ego_net_density (graphs_of_network: list[nx.Graph], nodes: "list" = None, hop: "int" = 2):
        if nodes is None:
            for j, graph in enumerate(graphs_of_network):
                print(bcolors.HEADER + f'Layer {j}: {graph}' + bcolors.ENDC)
                with tqdm(list(graph.nodes), unit=" Node") as t_nodes:
                    for node in t_nodes:
                        if graph.nodes[node]["ego_density"] is None:
                            ego_graph = nx.single_source_shortest_path_length(graph, node, cutoff=hop)
                            ego_graph_nodes = len(ego_graph)
                            ego_graph_edges = Network_Node_Centrality.get_sum_of_ndoes_degree(graph, list(ego_graph.keys()))
                            if ego_graph_nodes > 1:
                                ego_graph_density = ego_graph_edges / (ego_graph_nodes * (ego_graph_nodes - 1))
                            else:
                                ego_graph_density = 0
                            
                            node_ego_density = ((ego_graph_density * graph.nodes[node]['vote_power'])
                                                    * ego_graph_nodes)
                            graph.nodes[node]['ego_density'] = node_ego_density
        else:
            with tqdm(nodes, unit=" Node") as t_nodes:
                    for node in t_nodes:
                        for graph in graphs_of_network:
                            if node in graph and graph.nodes[node]["ego_density"] is None:
                                ego_graph = nx.single_source_shortest_path_length(graph, node, cutoff=hop)
                                ego_graph_nodes = len(ego_graph)
                                ego_graph_edges = Network_Node_Centrality.get_sum_of_ndoes_degree(graph, list(ego_graph.keys()))
                                if ego_graph_nodes > 1:
                                    ego_graph_density = ego_graph_edges / (ego_graph_nodes * (ego_graph_nodes - 1))
                                else:
                                    ego_graph_density = 0

                                node_ego_density = ((ego_graph_density * graph.nodes[node]['vote_power'])
                                                        * ego_graph_nodes)
                                graph.nodes[node]['ego_density'] = node_ego_density
        pass

    @staticmethod
    def ego_degree(graphs_of_network: list[nx.Graph], nodes: "list" = None, hop: "int" = 2):
        if nodes is None:
            for j, graph in enumerate(graphs_of_network):
                print(bcolors.HEADER + f'Layer {j}: {graph}' + bcolors.ENDC)
                with tqdm(list(graph.nodes), unit=" Node") as t_nodes:
                    for node in t_nodes:
                        if graph.nodes[node]["ego_degree"] is None:
                            ego_nodes = nx.single_source_shortest_path_length(graph, node, cutoff=hop)
                            degree_list = graph.degree(list(ego_nodes.keys()))
                            current_node_degree = degree_list[node]
                            score = current_node_degree
                            for k, v in degree_list:
                                if k != node:
                                    if current_node_degree != v:
                                        score += ((current_node_degree - v) * (1 / ego_nodes[k]))
                                    else:
                                        score += (math.sqrt(v) * (1 / ego_nodes[k]))
                            graph.nodes[node]['ego_degree'] = score
        else:
            with tqdm(nodes, unit=" Node") as t_nodes:
                    for node in t_nodes:
                        for graph in graphs_of_network:
                            if node in graph and graph.nodes[node]["ego_degree"] is None:
                                ego_nodes = nx.single_source_shortest_path_length(graph, node, cutoff=hop)
                                degree_list = graph.degree(list(ego_nodes.keys()))
                                current_node_degree = degree_list[node]
                                score = current_node_degree
                                for k, v in degree_list:
                                    if k != node:
                                        if current_node_degree != v:
                                            score += ((current_node_degree - v) * (1 / ego_nodes[k]))
                                        else:
                                            score += (math.sqrt(v) * (1 / ego_nodes[k]))
                                graph.nodes[node]['ego_degree'] = score
        pass

    @staticmethod
    def ego_k_shell(graphs_of_network: list[nx.Graph], nodes: "list" = None, hop: "int" = 2):
        if nodes is None:
            for j, graph in enumerate(graphs_of_network):
                print(bcolors.HEADER + f'Layer {j}: {graph}' + bcolors.ENDC)
                with tqdm(list(graph.nodes), unit=" Node") as t_nodes:
                    for node in t_nodes:
                        if graph.nodes[node]["ego_k_shell"] is None:
                            ego_nodes = nx.single_source_shortest_path_length(graph, node, cutoff=hop)
                            node_k_shell = graph.nodes[node]["k_shell"]
                            node_k_shell_itr = graph.nodes[node]["k_shell_itr"]
                            node_cc = 2 / (graph.nodes[node]["clustering"] + 1)
                            node_kss = ((node_k_shell * math.sqrt(node_k_shell)) * node_k_shell_itr) * node_cc
                            graph.nodes[node]['kss'] = node_kss
                            score = 0
                            for k, v in ego_nodes.items():
                                if k != node:
                                    neighbor_k_shell = graph.nodes[k]["k_shell"]
                                    neighbor_k_shell_itr = graph.nodes[k]["k_shell_itr"]
                                    neighbor_cc = 2 / (graph.nodes[k]["clustering"] + 1)
                                    neighbor_kss = ((neighbor_k_shell * math.sqrt(neighbor_k_shell)) * neighbor_k_shell_itr) * neighbor_cc
                                    if node_kss != neighbor_kss: 
                                        score += (node_kss - neighbor_kss) * (1 / v)
                                    else:
                                        score += (math.sqrt(node_kss) * (1 / v))
                            norm_val = node_kss * (graph.degree(node) * node_cc)
                            score += norm_val
                            graph.nodes[node]['ego_k_shell'] = score
        else:
            with tqdm(nodes, unit=" Node") as t_nodes:
                    for node in t_nodes:
                        for graph in graphs_of_network:
                            if node in graph and graph.nodes[node]["ego_k_shell"] is None:
                                ego_nodes = nx.single_source_shortest_path_length(graph, node, cutoff=hop)
                                node_k_shell = graph.nodes[node]["k_shell"]
                                node_k_shell_itr = graph.nodes[node]["k_shell_itr"]
                                node_cc = 2 / (graph.nodes[node]["clustering"] + 1)
                                node_kss = ((node_k_shell * math.sqrt(node_k_shell)) * node_k_shell_itr) * node_cc
                                graph.nodes[node]['kss'] = node_kss
                                score = 0
                                for k, v in ego_nodes.items():
                                    if k != node:
                                        neighbor_k_shell = graph.nodes[k]["k_shell"]
                                        neighbor_k_shell_itr = graph.nodes[k]["k_shell_itr"]
                                        neighbor_cc = graph.nodes[k]["clustering"]
                                        if neighbor_cc is None:
                                            neighbor_cc = nx.clustering(graph, k)
                                            graph.nodes[k]["clustering"] = neighbor_cc
                                        neighbor_cc = 2 / (neighbor_cc + 1)
                                        neighbor_kss = ((neighbor_k_shell * math.sqrt(neighbor_k_shell)) * neighbor_k_shell_itr) * neighbor_cc
                                        if node_kss != neighbor_kss: 
                                            score += (node_kss - neighbor_kss) * (1 / v)
                                        else:
                                            score += (math.sqrt(node_kss) * (1 /v))
                                
                                norm_val = node_kss * (graph.degree(node) * node_cc)
                                score += norm_val
                                graph.nodes[node]['ego_k_shell'] = score
        pass

    @staticmethod
    def ego_degree_mean(graphs_of_network: list[nx.Graph], nodes: "list" = None, hop: "int" = 2):
        if nodes is None:
            for j, graph in enumerate(graphs_of_network):
                print(bcolors.HEADER + f'Layer {j}: {graph}' + bcolors.ENDC)
                with tqdm(list(graph.nodes), unit=" Node") as t_nodes:
                    for node in t_nodes:
                        if graph.nodes[node]["ego_degree_mean"] is None:
                            neighbors_list = nx.single_source_shortest_path_length(graph, node, cutoff=hop)
                            del neighbors_list[node]
                            score = 0
                            for k, v in neighbors_list.items():
                                score += (graph.degree(k) * (1 / v))
                            score += (graph.degree(node) * (2 / (1 + graph.nodes[node]['clustering'])))
                            graph.nodes[node]['ego_degree_mean'] = score
        else:
            with tqdm(nodes, unit=" Node") as t_nodes:
                    for node in t_nodes:
                        for graph in graphs_of_network:
                            if node in graph and graph.nodes[node]["ego_degree_mean"] is None:
                                neighbors_list = nx.single_source_shortest_path_length(graph, node, cutoff=hop)
                                del neighbors_list[node]
                                score = 0
                                for k, v in neighbors_list.items():
                                    score += (graph.degree(k) * (1 / v))
                                score += (graph.degree(node) * (2 / (1 + graph.nodes[node]['clustering'])))
                                graph.nodes[node]['ego_degree_mean'] = score
        pass

    @staticmethod
    def get_nodes_centrality(network_infos_writer_object:object,
                             graphs_of_network:list[nx.Graph], nodes: list[int | str],
                             hop: int = 2) -> None:
            if nodes is None:
                for j, graph in enumerate(graphs_of_network):
                    if graph.number_of_nodes() > 0:
                        print(f'\nLayer {j}: {graph}')
                        counter = 0
                        node_list_to_write = []
                        node_counter = 0
                        Network_Node_Centrality.cluster_coefficient([graph], None)
                        with tqdm(list(graph.nodes), unit=" Node") as t_nodes:
                            for node in t_nodes:
                                counter += 1
                                node_counter += 1
                                node_list_to_write.append((j, node))
                                if graph.nodes[node]["nip"] is None:
                                    node_degree = graph.degree(node)
                                    neighbors_degree_cc = Network_Node_Centrality.get_neighbors_degree_cc(graph, node)
                                    node_cc = graph.nodes[node]["clustering"] # clustering coefficient
                                    graph.nodes[node]["nip"] = node_degree + (neighbors_degree_cc * (2 / (1 + node_cc)))
                                
                                if graph.nodes[node]["sombor_index"] is None:
                                    neighbors_list = list(graph.neighbors(node))
                                    curent_node_degree = graph.degree(node)
                                    sombor_index_score = 0
                                    for neighbor in neighbors_list:
                                        sombor_index_score += math.sqrt(
                                            (curent_node_degree ** 2) + (graph.degree(neighbor) ** 2))
                                    graph.nodes[node]['sombor_index'] = sombor_index_score
                                
                                if (graph.nodes[node]["ego_density"] is None or
                                    graph.nodes[node]["ego_degree"] is None or
                                    graph.nodes[node]["ego_k_shell"] is None or
                                    graph.nodes[node]["ego_degree_mean"] is None):
                                    ego_graph = nx.single_source_shortest_path_length(graph, node, cutoff=hop)

                                if graph.nodes[node]["ego_density"] is None:
                                    ego_graph_nodes = len(ego_graph)
                                    ego_graph_edges = Network_Node_Centrality.get_sum_of_ndoes_degree(graph, list(ego_graph.keys()))
                                    if ego_graph_nodes > 1:
                                        ego_graph_density = ego_graph_edges / (ego_graph_nodes * (ego_graph_nodes - 1))
                                    else:
                                        ego_graph_density = 0
                                    
                                    node_ego_density = ((ego_graph_density * graph.nodes[node]['vote_power'])
                                                            * ego_graph_nodes)
                                    graph.nodes[node]['ego_density'] = node_ego_density
                                
                                if graph.nodes[node]["ego_degree"] is None:
                                    degree_list = graph.degree(list(ego_graph.keys()))
                                    current_node_degree = degree_list[node]
                                    score = current_node_degree
                                    for k, v in degree_list:
                                        if k != node:
                                            if current_node_degree != v:
                                                score += ((current_node_degree - v) * (1 / ego_graph[k]))
                                            else:
                                                score += (math.sqrt(v) * (1 / ego_graph[k]))
                                    graph.nodes[node]['ego_degree'] = score
                                
                                if graph.nodes[node]["ego_k_shell"] is None:
                                    node_k_shell = graph.nodes[node]["k_shell"]
                                    node_k_shell_itr = graph.nodes[node]["k_shell_itr"]
                                    node_cc = 2 / (graph.nodes[node]["clustering"] + 1)
                                    node_kss = ((node_k_shell * math.sqrt(node_k_shell)) * node_k_shell_itr) * node_cc
                                    graph.nodes[node]['kss'] = node_kss
                                    score = 0
                                    for k, v in ego_graph.items():
                                        if k != node:
                                            neighbor_k_shell = graph.nodes[k]["k_shell"]
                                            neighbor_k_shell_itr = graph.nodes[k]["k_shell_itr"]
                                            neighbor_cc = 2 / (graph.nodes[k]["clustering"] + 1)
                                            neighbor_kss = ((neighbor_k_shell * math.sqrt(neighbor_k_shell)) * neighbor_k_shell_itr) * neighbor_cc
                                            if node_kss != neighbor_kss: 
                                                score += (node_kss - neighbor_kss) * (1 / v)
                                            else:
                                                score += (math.sqrt(node_kss) * (1 / v))
                                    norm_val = node_kss * (graph.degree(node) * node_cc)
                                    score += norm_val
                                    graph.nodes[node]['ego_k_shell'] = score
                                
                                if graph.nodes[node]["ego_degree_mean"] is None:
                                    del ego_graph[node]
                                    score = 0
                                    for k, v in ego_graph.items():
                                        score += (graph.degree(k) * (1 / v))
                                    score += (graph.degree(node) * (2 / (1 + graph.nodes[node]['clustering'])))
                                    graph.nodes[node]['ego_degree_mean'] = score

                            if ((counter >= 1000 and j >= (len(graphs_of_network) - 1)) or
                                    (node_counter >= len(t_nodes) and j >= (len(graphs_of_network) - 1))):
                                network_infos_writer_object.write_temp_network_nodes_info_csv(graphs_of_network, node_list_to_write)
                                node_list_to_write = []
                                counter = 0

            elif len(nodes) > 0:
                    with tqdm(nodes, unit=" Node") as t_nodes:
                        counter = 0
                        node_counter = 0
                        node_list_to_write = []
                        Network_Node_Centrality.cluster_coefficient(graphs_of_network, nodes)
                        for node in t_nodes:
                            node_counter += 1
                            for j, graph in enumerate(graphs_of_network):
                                if node in graph:
                                    counter += 1
                                    node_list_to_write.append((j, node))
                                    if graph.nodes[node]["nip"] is None:
                                        node_degree = graph.degree(node)
                                        neighbors_degree_cc = Network_Node_Centrality.get_neighbors_degree_cc(graph, node)
                                        node_cc = graph.nodes[node]["clustering"] # clustering coefficient
                                        graph.nodes[node]["nip"] = node_degree + (neighbors_degree_cc * (2 / (1 + node_cc)))
                                    if graph.nodes[node]["sombor_index"] is None:
                                        neighbors_list = list(graph.neighbors(node))
                                        curent_node_degree = graph.degree(node)
                                        sombor_index_score = 0
                                        for neighbor in neighbors_list:
                                            sombor_index_score += math.sqrt(
                                                (curent_node_degree ** 2) + (graph.degree(neighbor) ** 2))
                                        graph.nodes[node]['sombor_index'] = sombor_index_score

                                    if (graph.nodes[node]["ego_density"] is None or
                                        graph.nodes[node]["ego_degree"] is None or
                                        graph.nodes[node]["ego_k_shell"] is None or
                                        graph.nodes[node]["ego_degree_mean"] is None):
                                        ego_graph = nx.single_source_shortest_path_length(graph, node, cutoff=hop)

                                    if graph.nodes[node]["ego_density"] is None:
                                        ego_graph_nodes = len(ego_graph)
                                        ego_graph_edges = Network_Node_Centrality.get_sum_of_ndoes_degree(graph, list(ego_graph.keys()))
                                        if ego_graph_nodes > 1:
                                            ego_graph_density = ego_graph_edges / (ego_graph_nodes * (ego_graph_nodes - 1))
                                        else:
                                            ego_graph_density = 0

                                        node_ego_density = ((ego_graph_density * graph.nodes[node]['vote_power'])
                                                                * ego_graph_nodes)
                                        graph.nodes[node]['ego_density'] = node_ego_density
                                    
                                    if graph.nodes[node]["ego_degree"] is None:
                                        degree_list = graph.degree(list(ego_graph.keys()))
                                        current_node_degree = degree_list[node]
                                        score = current_node_degree
                                        for k, v in degree_list:
                                            if k != node:
                                                if current_node_degree != v:
                                                    score += ((current_node_degree - v) * (1 / ego_graph[k]))
                                                else:
                                                    score += (math.sqrt(v) * (1 / ego_graph[k]))
                                        graph.nodes[node]['ego_degree'] = score
                                    
                                    if graph.nodes[node]["ego_k_shell"] is None:
                                        node_k_shell = graph.nodes[node]["k_shell"]
                                        node_k_shell_itr = graph.nodes[node]["k_shell_itr"]
                                        node_cc = 2 / (graph.nodes[node]["clustering"] + 1)
                                        node_kss = ((node_k_shell * math.sqrt(node_k_shell)) * node_k_shell_itr) * node_cc
                                        graph.nodes[node]['kss'] = node_kss
                                        score = 0
                                        for k, v in ego_graph.items():
                                            if k != node:
                                                neighbor_k_shell = graph.nodes[k]["k_shell"]
                                                neighbor_k_shell_itr = graph.nodes[k]["k_shell_itr"]
                                                neighbor_cc = graph.nodes[k]["clustering"]
                                                if neighbor_cc is None:
                                                    neighbor_cc = nx.clustering(graph, k)
                                                    graph.nodes[k]["clustering"] = neighbor_cc
                                                neighbor_cc = 2 / (neighbor_cc + 1)
                                                neighbor_kss = ((neighbor_k_shell * math.sqrt(neighbor_k_shell)) * neighbor_k_shell_itr) * neighbor_cc
                                                if node_kss != neighbor_kss: 
                                                    score += (node_kss - neighbor_kss) * (1 / v)
                                                else:
                                                    score += (math.sqrt(node_kss) * (1 /v))
                                        
                                        norm_val = node_kss * (graph.degree(node) * node_cc)
                                        score += norm_val
                                        graph.nodes[node]['ego_k_shell'] = score

                                    if node in graph and graph.nodes[node]["ego_degree_mean"] is None:
                                        del ego_graph[node]
                                        score = 0
                                        for k, v in ego_graph.items():
                                            score += (graph.degree(k) * (1 / v))
                                        score += (graph.degree(node) * (2 / (1 + graph.nodes[node]['clustering'])))
                                        graph.nodes[node]['ego_degree_mean'] = score

                                    if ((counter >= 1000 and j >= (len(graphs_of_network) - 1)) or
                                         (node_counter >= len(nodes) and j >= (len(graphs_of_network) - 1))):
                                        network_infos_writer_object.write_temp_network_nodes_info_csv(graphs_of_network, node_list_to_write)
                                        node_list_to_write = []
                                        counter = 0
            else:
                print("Nodes list is empty!")
            pass


    pass