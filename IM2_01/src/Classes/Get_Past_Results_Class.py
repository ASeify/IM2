# version 1.5.1
import sys
import re
import os
import pandas as pd
import ast
import networkx as nx
import csv
import json


class Get_Past_Results:

    def __init__(self, network_path: str, network_name: str) -> None:
        self.past_result_paths = []
        self.network_name = network_name
        regex = r"\b[0-9]{4}+\ [0-9]{1,2}+\ [0-9]{1,2}+\_[0-9]{1,2}"  # +\ [0-9]{1,2}+\ [0-9]{1,2}\b'

        temp_past_result_paths = [
            x[0] for x in os.walk(str(network_path + network_name))
        ]
        for item in temp_past_result_paths:
            item = item.replace("\\", "/")
            if item != str(network_path + network_name):
                dir_name = item.rsplit("/")[-1]
                if re.fullmatch(regex, dir_name):
                    self.past_result_paths.append(item)       

    def get_past_SIR_results_from_temp_file(self, beta: float, landa: float, epoch: float):
        nodes_result_sir = {}
        past_SIR_result_files = []
        result_file_name = str(
            self.network_name
            + "_beta=" + str(beta)
            + "_landa=" + str(landa)
            + "_epoch=" + str(epoch)
            + " temp SIR.csv"
        )
        for item in self.past_result_paths:
            file_flag = os.path.isfile(str(item + "/" + result_file_name))
            if file_flag:
                past_SIR_result_files.append(str(item + "/" + result_file_name))
        for result_file in past_SIR_result_files:
            resulst = pd.read_csv(result_file).values
            for layer, node, sir in resulst:
                node = str(int(node))
                if not (node in nodes_result_sir):
                    nodes_result_sir[node] = {}
                nodes_result_sir[node][int(layer)] = sir
        return nodes_result_sir

    def get_past_SIR_results(self, beta: float, landa: float, epoch: float):
        nodes_result_sir = []
        finded_result_file = []
        i = len(self.past_result_paths) - 1
        while i >= 0:
            files_list = os.listdir(self.past_result_paths[i])
            if len(files_list) > 0:
                for item in files_list:
                    params_info = item.rsplit(".")
                    result_file_info = ''
                    if len(params_info) > 2:
                        for j, jitem in enumerate(params_info):
                            if j >= (len(params_info) - 1):
                                break
                            else:
                                if j >= (len(params_info) - 2):
                                    result_file_info += jitem
                                else:
                                    result_file_info += (jitem + '.')
                    params_info = result_file_info.split("_")
                    if (params_info[-1].split("=")[-1] == str(epoch) and
                        params_info[-2].split("=")[-1] == str(landa) and
                        params_info[-3].split("=")[-1] == str(beta)):
                                finded_result_file.append(self.past_result_paths[i] + "/" + item)
            i -= 1
        if len(finded_result_file)  <=  0:
            return False, finded_result_file
        else:
            for result_file in finded_result_file:
                resulst = pd.read_csv(result_file)
                attributes_title = resulst.keys()
                for item in resulst.values:
                    for i, attribute in enumerate(attributes_title):
                        if not(attribute == 'node_id') and not(attribute == 'Mean'):
                            layer_id = int(attribute.split(" ")[-1])
                            if item[i] != 0:
                                nodes_result_sir.append((layer_id, str(int(item[0])), item[i]))         
            return True, nodes_result_sir

    def get_past_node_att_results(self, graphs_of_network:list[nx.Graph]):
        nodes_result_sir = {}
        finded_result_file = None
        i = len(self.past_result_paths) - 1
        while i >= 0:
            files_list = os.listdir(self.past_result_paths[i])
            if len(files_list) > 0:
                for item in files_list:
                    if item.rsplit(".")[-2].split(" ")[-1] == "graph":
                        finded_result_file = self.past_result_paths[i] + "/" + item
                        break
                if not (finded_result_file is None):
                    break
            i -= 1
        if finded_result_file is None:
            return False, nodes_result_sir
        else:
            resulst = pd.read_csv(finded_result_file)
            attributes_title = resulst.keys()
            for item in resulst.values:
                for i, node_attribute in enumerate(attributes_title):
                    if node_attribute != 'layer_id' and node_attribute != 'node_id' :
                        res = str(item[i]).isdigit()
                        if res == True:
                            item_value = int(item[i])
                        else:
                            item_value = float(item[i])
                        graphs_of_network[int(item[0])].nodes[str(int(item[1]))][str(node_attribute)] = item_value
                    if node_attribute == 'SIR':
                        if not (str(int(item[1])) in nodes_result_sir):
                            nodes_result_sir[str(int(item[1]))] = {}    
                        nodes_result_sir[str(int(item[1]))][int(item[0])] = float(item[i])
            return True, nodes_result_sir

    def get_past_node_SIR_results_from_graph_file(self, graphs_of_network:list[nx.Graph]):
        nodes_result_sir = {}
        finded_result_file = None
        i = len(self.past_result_paths) - 1
        while i >= 0:
            files_list = os.listdir(self.past_result_paths[i])
            if len(files_list) > 0:
                for item in files_list:
                    if item.rsplit(".")[-2].split(" ")[-1] == "graph":
                        finded_result_file = self.past_result_paths[i] + "/" + item
                        break
                if not (finded_result_file is None):
                    break
            i -= 1
        if finded_result_file is None:
            return False, nodes_result_sir
        else:
            resulst = pd.read_csv(finded_result_file)
            attributes_title = resulst.keys()
            for item in resulst.values:
                for i, node_attribute in enumerate(attributes_title):
                    if node_attribute == 'SIR':
                        item_value = float(item[i])
                        graphs_of_network[int(item[0])].nodes[str(int(item[1]))][str(node_attribute)] = item_value
                        if not (str(int(item[1])) in nodes_result_sir):
                            nodes_result_sir[str(int(item[1]))] = {}    
                        nodes_result_sir[str(int(item[1]))][int(item[0])] = float(item[i])
            return True, nodes_result_sir

    def get_past_layer_att_results(self, graphs_of_network:list[nx.Graph]):
        finded_result_file = None
        i = len(self.past_result_paths) - 1
        while i >= 0:
            files_list = os.listdir(self.past_result_paths[i])
            if len(files_list) > 0:
                for item in files_list:
                    if item.rsplit(".")[-2].split(" ")[-1] == "layers":
                        finded_result_file = self.past_result_paths[i] + "/" + item
                        break
                if not (finded_result_file is None):
                    break
            i -= 1
        
        if finded_result_file is None:
            return False
        else:
            resulst = pd.read_csv(finded_result_file)
            attributes_title = resulst.keys()
            for item in resulst.values:
                for i, layer_attribute in enumerate(attributes_title):
                    if layer_attribute != 'id':
                        item_value = item[i]
                        if type(item_value) is str and "[" in item_value:
                            item_value = ast.literal_eval(item_value)
                        graphs_of_network[int(item[0])].graph[layer_attribute] = item_value
            return True
    
    def get_past_layer_att_results_from_graph_file(self, graphs_of_network:list[nx.Graph]):
        finded_result_file = None
        i = len(self.past_result_paths) - 1
        while i >= 0:
            files_list = os.listdir(self.past_result_paths[i])
            if len(files_list) > 0:
                for item in files_list:
                    if item.rsplit(".")[-2].split(" ")[-1] == "graph":
                        finded_result_file = self.past_result_paths[i] + "/" + item
                        break
                if not (self.past_result_paths is None):
                    break
            i -= 1
        if finded_result_file is None:
            return False
        else:
            resulst = pd.read_csv(finded_result_file)
            attributes_title = resulst.keys()
            layer_attributes = ['layer_density', 'layer_degree_histogram', 'layer_edge_weight',
                          'layer_sombor_index', 'layer_nodes_weight', 'layer_k_shell_weight']
            nodes_attributes = ['degree', 'weight', 'k_shell', 'k_shell_itr', 'nip',
                                'sombor_index', 'ego_density', 'ego_degree', 'ego_k_shell',
                                  'ego_degree_mean', 'kss', 'vote_power', 'clustering']
            layer_id = -1
            for item in resulst.values:
                if int(item[0]) != layer_id:
                    for i, node_attribute in enumerate(attributes_title):
                        if node_attribute == 'layer_id':
                            layer_id = int(item[i])
                        if (node_attribute != 'layer_id') and (not(node_attribute in nodes_attributes)):
                            if node_attribute in layer_attributes:
                                graphs_of_network[layer_id].graph[node_attribute] = float (item)
                            else:
                                return False
                        elif node_attribute == 'layer_id' and layer_id != int(item[i]):
                            layer_id = int(item[i])
            return True

    def read_layer_nodes_att_from_csv(graphs_of_network: list[nx.Graph], graph:nx.Graph, node_att:str, network_info:list[str]):
        files_path = network_info['path']+network_info['name']+'/' + node_att
        files_list = next(os.walk(files_path), (None, None, []))[2]
        finded_file_flag = False
        finded_file = None
        nodes_att = {}
        for item in files_list:
            if int(item.split('.')[0]) == graph.graph['id']:
                finded_file = item
                finded_file_flag = True
                break

        if finded_file_flag:
            with open(files_path+'/'+finded_file) as csv_file:
                reader = csv.reader(csv_file)
                nodes_att = dict(reader)

            for node, att in nodes_att.items():
                graphs_of_network[int(graph.graph['id'])].nodes[node][node_att] = float(att)
        return finded_file_flag, nodes_att

    def read_nodes_att_from_temp_graph(self, graphs_of_network:list[nx.Graph]):
        finded_result_file = []
        nodes_heve_result = []
        for path_item in self.past_result_paths:
            files_list = os.listdir(path_item)
            if len(files_list) > 0:
                for file_item in files_list:
                    file_name_items = file_item.rsplit(".")[-2].split(" ")
                    if file_name_items[-1] == "temp" and file_name_items[-2] == "graph":
                        finded_result_file.append(path_item + "/" + file_item)
                        break
        if len(finded_result_file) <= 0:
            return False, []
        else:
            for file_item in finded_result_file:
                resulst = pd.read_csv(file_item, skip_blank_lines=True, on_bad_lines='warn')
                attributes_title = resulst.keys()
                for item in resulst.values:
                    nodes_heve_result.append(str(int(item[1])))
                    for i, node_attribute in enumerate(attributes_title):
                        if node_attribute != 'layer_id' and node_attribute != 'node_id' :
                            res = str(item[i]).isdigit()
                            if res == True:
                                item_value = int(item[i])
                            else:
                                item_value = float(item[i])
                            graphs_of_network[int(item[0])].nodes[str(int(item[1]))][str(node_attribute)] = item_value
            return True, list(set(nodes_heve_result))
    
    def get_past_node_att_file(self):
        i = len(self.past_result_paths) - 1
        finded_result_file = None
        while i >= 0:
            files_list = os.listdir(self.past_result_paths[i])
            if len(files_list) > 0:
                for item in files_list:
                    if item.rsplit(".")[-2].split(" ")[-1] == "graph":
                        finded_result_file = self.past_result_paths[i] + "/" + item
                        break
                if not (self.past_result_paths is None):
                    break
            i -= 1
        if finded_result_file is None:
            return False
        else:
            return finded_result_file
        pass

    @staticmethod
    def load_results_from_file(root_path:os.path, file_name:str):
        load_flag = False
        info = None

        read_file = root_path + file_name + '.json'
        try:
            with open(read_file, 'r') as openfile:
                info = json.load(openfile)
            load_flag = True
        except Exception as e:
            print(e)
            load_flag = False
        return info, load_flag
    
    pass