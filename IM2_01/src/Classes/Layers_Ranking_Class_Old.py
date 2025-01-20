# version: 2.1.0

import math
from tqdm import tqdm
import networkx as nx
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if not (CURRENT_DIR in sys.path):
    sys.path.append(CURRENT_DIR)

from Bcolors_Class import Bcolors
bcolors = Bcolors()


class Layers_Ranking:

    @staticmethod
    def layers_density_weight(graphs_of_network:list[nx.Graph]):
        with tqdm(graphs_of_network, unit=" Layer") as t_graphs:
            layer_counter = 0
            for graph in t_graphs:
                if graph.number_of_nodes() > 0 and graph.number_of_edges() > 0:
                    t_graphs.set_description(f"Layer {layer_counter + 1}")
                    v = graph.number_of_nodes()
                    e = graph.number_of_edges()
                    if v > 0 :
                        layer_density = (2 * e) / (v * (v - 1))
                        graph.graph['layer_density'] = layer_density
                    else:
                        graph.graph['layer_density'] = 0
                layer_counter += 1
        pass

    @staticmethod
    def layers_degree_distribution(graphs_of_network:list[nx.Graph]):
        network_layers_count = len(graphs_of_network)
        i = 0
        pbar_layer = tqdm(total=network_layers_count, unit='Layer')
        while i < network_layers_count:
            degree_histogram = [0]
            pbar_layer.set_description(f"Layer {i + 1}")
            for node in graphs_of_network[i].nodes():
                node_degree = graphs_of_network[i].degree(node)
                if len(degree_histogram) <= node_degree:
                    while len(degree_histogram) <= node_degree:
                        degree_histogram.append(0)
                degree_histogram[node_degree] += 1

            v = nx.number_of_nodes(graphs_of_network[i])
            j, k = len(degree_histogram), 0
            layers_degree_distribution_weight = 0
            while j > k:
                if v > 0:
                    layers_degree_distribution_weight += ((degree_histogram[k] / v) * k)
                k += 1
            graphs_of_network[i].graph[ "layer_degree_histogram"] = round((layers_degree_distribution_weight), 7)
            pbar_layer.update(1)
            i += 1
        pbar_layer.close()
        pass

    @staticmethod
    def layers_edges_and_sombor_index(graphs_of_network:list[nx.Graph]):
        for j, graph in enumerate(graphs_of_network):
            if graph.number_of_nodes() > 0:
                print(bcolors.HEADER + f'Layer {j}: {graph}' + bcolors.ENDC)
                layer_sombor_index_weight = 0
                e = graph.number_of_edges()
                graph.graph["layer_edge_weight"] = (e / 100000)
                with tqdm(graph.edges, unit=" Edge") as t_edges:
                    for i, edge in enumerate(t_edges):
                        t_edges.set_description(f"Edge {i + 1}")
                        layer_edge_weight = round((math.sqrt(
                            (((graph.degree(edge[0])) ** 2)
                            + ((graph.degree(edge[1])) ** 2)))),7)
                        layer_sombor_index_weight += layer_edge_weight
                e = graph.number_of_edges()
                e *= (math.log(e, 2))
                if e == 0:
                    e = 1
                graph.graph["layer_sombor_index"] = round((layer_sombor_index_weight / e), 7)
            else:
                graph.graph["layer_sombor_index"] = 0
        pass

    @staticmethod
    def layers_nodes_weight(graphs_of_network:list[nx.Graph]):
        network_layers_count = len(graphs_of_network)
        q = 0
        pbar = tqdm(total=(network_layers_count), unit='Layer')
        while q < network_layers_count:
            if graphs_of_network[q].number_of_nodes() > 0:
                pbar.set_description(f"Layer {q + 1}")
                graphs_of_network[q].graph["layer_nodes_weight"] = (graphs_of_network[q].number_of_nodes() / 10000)
            q += 1
            pbar.update()
        pbar.close()
        pass

    @staticmethod
    def layers_k_shell_weight(graphs_of_network:list[nx.Graph]):
        network_layers_count = len(graphs_of_network)
        i = 0
        pbar = tqdm(total=(network_layers_count), unit='Layer')
        while i < network_layers_count:
            if graphs_of_network[i].number_of_nodes() > 0:
                pbar.set_description(f"Layer {i + 1}")
                k_shell_list = list(graphs_of_network[i].graph["k_shell_info"])
                v = nx.number_of_nodes(graphs_of_network[i])
                j = 0
                layer_k_shell_weight = 0
                while j < len(k_shell_list):
                    k = 0
                    while k < len(k_shell_list[j]):
                        layer_k_shell_weight += ((k_shell_list[j][k] / v) * ((j * math.sqrt(j)) * (k + 1)))
                        k += 1
                    j += 1
                norm_val = (math.sqrt(len(k_shell_list)))
                graphs_of_network[i].graph["layer_k_shell_weight"] = round((layer_k_shell_weight / norm_val), 7)
            i += 1
            pbar.update()
        pbar.close()
        pass

    pass
