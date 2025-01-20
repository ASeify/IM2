# version: 1.1.5
import numpy as np
import networkx as nx
from tqdm import tqdm
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if not (CURRENT_DIR in sys.path):
    sys.path.append(CURRENT_DIR)

from Bcolors_Class import Bcolors
bcolors = Bcolors()

class SIR_Diffusion_Model:
    @staticmethod
    def SIR_multilayer_model(
        layers_of_network: list["nx.Graph"],
        beta: "float",
        landa: "float",
        iteration: "int",
        bias: int = None,
    ) -> list[dict]:
        layers_nodes_infect_scale = []
        SIR_multilayer_model_object = SIR_Diffusion_Model()
        for layer in layers_of_network:
            if layer.number_of_nodes() > 0:
                print(bcolors.HEADER + f'Layer {layer.graph["id"]}: {layer}' + bcolors.ENDC)
                layers_nodes_infect_scale.append(
                    SIR_multilayer_model_object.SIR_monoplex_model(
                        layer, beta, landa, iteration, bias
                    )
                )
        return layers_nodes_infect_scale

    @staticmethod
    def SIR_multilayer_with_seed_set_model(
        layers_of_network: "nx.Graph",
        seed_sets_list: "list",
        beta: "float",
        landa: "float",
        iteration: "int",
        bias: int = None,
    ) -> list[float]:
        layers_seed_set_infect_scale = []
        SIR_multilayer_model_object = SIR_Diffusion_Model()
        for layer in layers_of_network:
            if layer.number_of_nodes() > 0:
                print(bcolors.HEADER + f'Layer {layer.graph["id"]}: {layer}' + bcolors.ENDC)
                if bias is None:
                    layers_seed_set_infect_scale.append(
                        SIR_multilayer_model_object.SIR_monoplex_with_seed_set_model(
                            layer, seed_sets_list, beta, landa, iteration
                        )
                    )
                else:
                    layers_seed_set_infect_scale.append(
                        SIR_multilayer_model_object.biased_SIR_monoplex_with_seed_set_model(
                            layer, seed_sets_list, beta, landa, iteration, bias
                        )
                    )
        return layers_seed_set_infect_scale

    @staticmethod
    def SIR_monoplex_model(
        network: "nx.Graph",
        beta: "float",
        landa: "float",
        iteration: "int",
        bias: int = None,
    ) -> dict:
        nodes_infect_scale_list = {}
        SIR_monoplex_model_object = SIR_Diffusion_Model()
        pbar = tqdm(total=len(network.nodes()))
        pbar.colour = 'green'
        pbar.unit = ' Node'
        p_itr = 1
        infect_scale = 0
        for node in network.nodes():
            seed = set()
            seed.add(node)
            if bias is None:
                nodes_infect_scale_list[node] = (
                    SIR_monoplex_model_object.SIR_monoplex_with_seed_set_model(
                        network, seed, beta, landa, iteration
                    )
                )
            else:
                nodes_infect_scale_list[node] = (
                    SIR_monoplex_model_object.biased_SIR_monoplex_with_seed_set_model(
                        network, seed, beta, landa, iteration, bias
                    )
                )
            infect_scale += nodes_infect_scale_list[node]
            pbar.set_description(f'Infect scale mean {infect_scale / p_itr}')
            p_itr += 1
            pbar.update(1)
        pbar.close()
        return nodes_infect_scale_list

    @staticmethod
    def SIR_monoplex_with_seed_set_model(
        network: "nx.Graph",
        seed: "list",
        beta: "float",
        landa: "float",
        iteration: "int",
    ) -> float:
        infect_scale = 0  # for all of iterations
        n = iteration  # number of simulation iteration
        pbar = tqdm(total=n)
        pbar.colour = 'green'
        pbar.unit = ' Iteration'
        p_itr = 1
        while n > 0:
            nodes_label = {}  # label is one of [I, R]
            node_infect_scale = 0  # # for one of iterations
            infected_nodes_set = set()  # Seed set

            # all of the network nodes set on suseptible state
            # for node in network.nodes:
            #     nodes_label[node] = "s"

            for seed_node in seed:
                nodes_label[seed_node] = "i"  # Change seed node to infected state
                node_infect_scale += 1
                infected_nodes_set.add(seed_node)  # Add seed node to seed set

            while len(infected_nodes_set) > 0:  # While is an infected node teh network
                new_infected_nodes = set()  # Temp list for new infected nodes
                recoverd_nodes = set()  # Temp list for lastest recoverd nodes
                # Check seed nodes neighbors become infect or not
                # Check seed nodes change his state from infected to recoverd
                for infect_node in infected_nodes_set:
                    if infect_node in network.nodes():
                        # Get seed node neighbors in the network
                        neighbors_of_infect_node = list(network.neighbors(infect_node))
                        for item in neighbors_of_infect_node:
                            if not (item in nodes_label):
                                # Get infection chance (infection_chance in (0, 1))
                                infection_chance = np.random.random_sample()
                                if beta >= infection_chance:
                                    nodes_label[item] = "i"
                                    node_infect_scale += 1
                                    new_infected_nodes.add(item)
                    # Get recovery chance (infection_chance in (0, 1))
                    recovery_chance = np.random.random_sample()
                    if landa >= recovery_chance:
                        nodes_label[infect_node] = "r"
                        recoverd_nodes.add(infect_node)
                infected_nodes_set.update(
                    new_infected_nodes
                )  # Add new infected nodes to seed set
                infected_nodes_set -= (
                    recoverd_nodes  # Remove recoverd nodes from seed set
                )
            infect_scale += node_infect_scale
            n -= 1
            pbar.set_description(f'Infect scale {infect_scale / p_itr}')
            p_itr += 1
            pbar.update(1)
        pbar.close()
        return infect_scale / iteration

    @staticmethod
    def biased_SIR_monoplex_with_seed_set_model(
        network: "nx.Graph",
        seed: "list",
        beta: "float",
        landa: "float",
        iteration: "int",
        bias: int = 5,
    ) -> float:
        infect_scale = 0  # for all of iterations
        n = iteration  # number of simulation iteration
        pbar = tqdm(total=n)
        pbar.colour = 'green'
        pbar.unit = ' Iteration'
        p_itr = 1
        while n > 0:
            nodes_label = {}  # label is one of [I, R]
            node_infect_scale = 0  # # for one of iterations
            infected_nodes_set = set()  # Seed set

            # all of the network nodes set on suseptible state
            # for node in network.nodes:
            #     nodes_label[node] = "s"

            for seed_node in seed:
                nodes_label[seed_node] = "i"  # Change seed node to infected state
                node_infect_scale += 1
                infected_nodes_set.add(seed_node)  # Add seed node to seed set
            bias_counter = 0
            while bias > bias_counter and len(infected_nodes_set) > 0:
                bias_counter += 1
                new_infected_nodes = set()  # Temp list for new infected nodes
                recoverd_nodes = set()  # Temp list for lastest recoverd nodes
                # Check seed nodes neighbors become infect or not
                # Check seed nodes change his state from infected to recoverd
                for infect_node in infected_nodes_set:
                    # Get seed node neighbors in the network
                    neighbors_of_infect_node = list(network.neighbors(infect_node))
                    for item in neighbors_of_infect_node:
                        if not (item in nodes_label):
                            # Get infection chance (infection_chance in (0, 1))
                            infection_chance = np.random.random_sample()
                            if beta >= infection_chance:
                                nodes_label[item] = "i"
                                node_infect_scale += 1
                                new_infected_nodes.add(item)
                    # Get recovery chance (infection_chance in (0, 1))
                    recovery_chance = np.random.random_sample()
                    if landa >= recovery_chance:
                        nodes_label[infect_node] = "r"
                        recoverd_nodes.add(infect_node)
                infected_nodes_set.update(
                    new_infected_nodes
                )  # Add new infected nodes to seed set
                infected_nodes_set -= (
                    recoverd_nodes  # Remove recoverd nodes from seed set
                )
            infect_scale += node_infect_scale
            n -= 1
            pbar.set_description(f'Infect scale {"%.5f" % round(((infect_scale / p_itr) / network.number_of_nodes()), 5)}')
            p_itr += 1
            pbar.update(1)
        pbar.close()
        return infect_scale / iteration

    @staticmethod
    def synchronous_SIR_multilayer_with_seed_set_model(
        layers_of_network: list["nx.Graph"],
        seed: "list",
        beta: "float",
        landa: "float",
        iteration: "int",
        network_entier_nodes_list:list=None):
        infect_scale = 0  # for all of iterations
        n = iteration  # number of simulation iteration
        pbar = tqdm(total=n)        
        pbar.colour = 'green'
        pbar.unit = ' Iteration'
        p_itr = 1
        network_entier_nodes_coun = 1
        if not(network_entier_nodes_list is None):
            if len(network_entier_nodes_list) > 0:
                network_entier_nodes_coun = len(network_entier_nodes_list)
        while n > 0:
            nodes_label = {}  # label is one of [I, R]
            node_infect_scale = 0  # # for one of iterations
            infected_nodes_set = set()  # Seed set

            # all of the network nodes set on suseptible state
            # for node in network.nodes:
            #     nodes_label[node] = "s"

            for seed_node in seed:
                nodes_label[seed_node] = "i"  # Change seed node to infected state
                node_infect_scale += 1
                infected_nodes_set.add(seed_node)  # Add seed node to seed set
            while len(infected_nodes_set) > 0:
                new_infected_nodes = set()  # Temp list for new infected nodes
                recoverd_nodes = set()  # Temp list for lastest recoverd nodes
                # Check seed nodes neighbors become infect or not
                # Check seed nodes change his state from infected to recoverd
                for infect_node in infected_nodes_set:
                    for layer in layers_of_network:
                        if layer.number_of_nodes() > 0 and infect_node in layer.nodes():
                            # Get seed node neighbors in the network
                            neighbors_of_infect_node = list(layer.neighbors(infect_node))
                            for item in neighbors_of_infect_node:
                                if not (item in nodes_label):
                                    # Get infection chance (infection_chance in (0, 1))
                                    infection_chance = np.random.random_sample()
                                    if beta >= infection_chance:
                                        nodes_label[item] = "i"
                                        node_infect_scale += 1
                                        new_infected_nodes.add(item)
                    # Get recovery chance (infection_chance in (0, 1))
                    recovery_chance = np.random.random_sample()
                    if landa >= recovery_chance:
                        nodes_label[infect_node] = "r"
                        recoverd_nodes.add(infect_node)
                infected_nodes_set.update(
                    new_infected_nodes
                )  # Add new infected nodes to seed set
                infected_nodes_set -= (
                    recoverd_nodes  # Remove recoverd nodes from seed set
                )
            infect_scale += node_infect_scale
            n -= 1
            pbar.set_description(f'Infect scale {"%.5f" % round(((infect_scale / p_itr) / network_entier_nodes_coun), 5)}')
            p_itr += 1
            pbar.update(1)
        pbar.close()
        return infect_scale / iteration

    pass