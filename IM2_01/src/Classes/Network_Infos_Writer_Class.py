# version: 1.4.1
import os
import csv
from datetime import datetime
from pathlib import Path
import networkx as nx
import pandas as pd
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if not (CURRENT_DIR in sys.path):
    sys.path.append(CURRENT_DIR)

from Files_Handler_Class import Files_Handler
file_handler_obj = Files_Handler()

class Network_Infos_Writer:

    def __init__(self, network_path: "str", network_name: "str"):
        self.network_path = network_path
        self.network_name = network_name
        # self.network_name = network_name.replace(" ", "_")
        # Make a directory for network infos files (if not exist)
        self.path = file_handler_obj.make_dir(self.network_path, self.network_name)
        # get current datetime for create unique folsers name
        date_now = datetime.now()
        current_datetime_str = str(date_now.year) +" "+ str(date_now.month) +" "+ str(date_now.day)
        current_datetime_str += "_" + str(date_now.hour) # +" "+ str(date_now.minute) +" "+ str(date_now.second)
        self.current_datetime_str = current_datetime_str
        self.path = file_handler_obj.make_dir(self.path, self.current_datetime_str)

    def write_SIR_infos(
        self,
        node_fraction: float,
        beta: "float",
        landa: "float",
        epoch: "int",
        file_extension: "str",
        graphs_of_network: list[nx.Graph],
        nodes: list = None,
    ):
        if nodes is None:
            new_file_name = (self.network_name + "_fn=" + str(node_fraction) +
                          "_beta=" + str(beta) + "_landa=" + str(landa) + "_epoch=" + str(epoch))
            # Create or Ediet network SIR infos
            network_SIR_file_opened = open(str(self.path + new_file_name + file_extension),"w",)
            # Write in first  line of file SIR seted parameters
            network_SIR_file_opened.write(
                "fn=" + str(node_fraction) + ", beta=" + str(beta) +
                ", lambda=" + str(landa) + ", epoch=" + str(epoch) + "\n"
            )
            for j, graph in enumerate(graphs_of_network):
                if graph.number_of_nodes > 0:
                    network_SIR_file_opened.write(
                        "layer " + str(j) + " = "
                        + str(nx.get_node_attributes(graph, "SIR"))
                        + "\n"
                    )
            network_SIR_file_opened.close()  # Close opened file
        elif len(nodes) > 0:
            new_file_name = (self.network_name + "_fn=" + str(node_fraction) +
                          "_beta=" + str(beta) + "_landa=" + str(landa) + "_epoch=" + str(epoch))
            # Create or Ediet network SIR infos
            network_SIR_file_opened = open(str(self.path + new_file_name + file_extension),"w",)
            # Write in first  line of file SIR seted parameters
            network_SIR_file_opened.write(
                "fn=" + str(node_fraction) + ", beta=" + str(beta) +
                ", lambda=" + str(landa) + ", epoch=" + str(epoch) + "\n"
            )
            nodes_sir_list = []
            for graph in graphs_of_network:
                nodes_sir_list.append({})
            for node in nodes:
                for i, graph in enumerate(graphs_of_network):
                    if graph.number_of_nodes() > 0 and node in graph:
                        nodes_sir_list[i][node] = graph.nodes[node]["SIR"]
            # print(nodes_sir_list)
            for j, item in enumerate(nodes_sir_list):
                if len(item) > 0:
                    network_SIR_file_opened.write(
                        "layer " + str(j) + " = " + str(item) + "\n"
                    )
            network_SIR_file_opened.close()  # Close opened file
        else:
            print("Nodes list is empty!")
        pass

    def wirte_layers_mean_SIR_infos_csv(
        self,
        node_fraction: float,
        beta: "float",
        landa: "float",
        epoch: "int",
        graphs_of_network: list[nx.Graph],
        nodes: list = None,
    ):

        if nodes is None:
            new_file_name = (self.network_name + "_fn=" + str(node_fraction) +
                          "_beta=" + str(beta) + "_landa=" + str(landa) + "_epoch=" + str(epoch))

            csv_file = open(str(self.path + new_file_name+ ".csv"), "w", newline="")
            csv_writer = csv.writer(csv_file)
            layers_head_lable = []
            layers_head_lable.append("node_id")
            layer_count = len(graphs_of_network)
            network_entier_nodes = []
            for i in range(layer_count):
                if graphs_of_network[i].number_of_nodes() > 0:
                    layers_head_lable.append("layer " + str(i))
            layers_head_lable.append("Mean")
            csv_writer.writerow(layers_head_lable)
            for i, graph in enumerate(graphs_of_network):
                if graph.number_of_nodes() > 0:
                    network_entier_nodes += list(graph.nodes())

            network_entier_nodes = list(set(network_entier_nodes))
            for node in network_entier_nodes:
                infos = [node]
                sum_of_sir = 0
                layer_count = 0
                for graph in graphs_of_network:
                    if graph.number_of_nodes() > 0:
                        layer_count += 1
                        if node in graph:
                            sir_value = graph.nodes[node]['SIR']
                            infos.append(sir_value)
                            sum_of_sir += sir_value
                        else:
                            infos.append(0)
                infos.append(sum_of_sir / layer_count)
                csv_writer.writerow(infos)
            csv_file.close()  # Close opened file
        elif len(nodes) > 0:
            new_file_name = (self.network_name + "_fn=" + str(node_fraction) +
                          "_beta=" + str(beta) + "_landa=" + str(landa) + "_epoch=" + str(epoch))

            csv_file = open(str(self.path + new_file_name+ ".csv"), "w", newline="")
            csv_writer = csv.writer(csv_file)
            layers_head_lable = []
            layers_head_lable.append("node_id")
            layer_count = len(graphs_of_network)
            network_entier_nodes = []
            for i in range(layer_count):
                if graphs_of_network[i].number_of_nodes() > 0:
                    layers_head_lable.append("layer " + str(i))
            layers_head_lable.append("Mean")
            csv_writer.writerow(layers_head_lable)
            for node in nodes:
                infos = [node]
                sum_of_sir = 0
                layer_count = 0
                for graph in graphs_of_network:
                    if graph.number_of_nodes() > 0:
                        layer_count += 1
                        if node in graph:
                            sir_value = graph.nodes[node]['SIR']
                            infos.append(sir_value)
                            sum_of_sir += sir_value
                        else:
                            infos.append(0)
                infos.append(sum_of_sir / layer_count)
                csv_writer.writerow(infos)
            csv_file.close()  # Close opened file
        else:
            print("Nodes list is empty!")
        pass

    def write_network_nodes_info(
        self,
        node_fraction: float,
        file_extension: "str",
        graphs_of_network: list[nx.Graph],
        nodes: list = None,
    ):

        if nodes is None:
            network_file_opened = open(str(self.path + self.network_name + "_fn=" + str(node_fraction) + file_extension), "w")
            # Write network seted parameters in first line
            network_file_opened.write("fn=" + str(node_fraction) + "\n")
            for i, graph in enumerate(graphs_of_network):
                for node in graph:
                    if graph.number_of_nodes() > 0:
                        # Write current node infos in network info file
                        network_file_opened.write(
                            "layer : " + str(i)
                            + " , node_id : " + str(node)
                            + " , "
                            + str(graph.nodes[node])
                            + "\n")
            network_file_opened.close()  # Close opened fiel
        elif len(nodes) > 0:
            network_file_opened = open(str(self.path + self.network_name + "_fn=" + str(node_fraction) + file_extension), "w")
            # Write network seted parameters in first line
            network_file_opened.write("fn=" + str(node_fraction) + "\n")
            for node in nodes:
                for i, graph in enumerate(graphs_of_network):
                    if graph.number_of_nodes() > 0 and node in graph:
                        # Write current node infos in network info file
                        network_file_opened.write(
                            "layer : " + str(i)
                            + " , node_id : " + str(node)
                            + " , "
                            + str(graph.nodes[node])
                            + "\n")
            network_file_opened.close()  # Close opened fiel
        else:
            print("Nodes list is empty!")
        pass

    def write_network_nodes_info_csv(
        self,
        node_fraction: float,
        graphs_of_network: list[nx.Graph],
        nodes: list = None,
    ):
        
        if nodes is None:
            # Crete or edite network infos file
            network_file_opened = open(
                str(self.path + self.network_name + "_fn=" + str(node_fraction) + " graph" + ".csv"), "w", newline="")
            attributes_list = ["layer_id", "node_id"]
            node = None
            temp_graph = None
            for g in graphs_of_network:
                if g.number_of_nodes() > 0:
                    node = list(g.nodes)[0]
                    temp_graph = g
                    break
            attributes_list += list(temp_graph.nodes[node].keys())
            del temp_graph
            csv_writer = csv.writer(network_file_opened)
            csv_writer.writerow(attributes_list)
            for i, graph in enumerate(graphs_of_network):
                for node in graph:
                    if graph.number_of_nodes() > 0:
                        values_list = [str(i), str(node)]
                        values_list += list(graph.nodes[node].values())[:]
                        csv_writer.writerow(values_list)
            network_file_opened.close()
        elif len(nodes) > 0:
            # Crete or edite network infos file
            network_file_opened = open(
                str(self.path + self.network_name + "_fn=" + str(node_fraction) + " graph" + ".csv"), "w", newline="")
            attributes_list = ["layer_id", "node_id"]
            node = nodes[0]
            temp_graph = None
            for g in graphs_of_network:
                if g.number_of_nodes() > 0:
                    if node in g:
                        if node in g:
                            temp_graph = g
                            break
            attributes_list += list(temp_graph.nodes[node].keys())
            del temp_graph
            csv_writer = csv.writer(network_file_opened)
            csv_writer.writerow(attributes_list)
            for node in nodes:
                for i, graph in enumerate(graphs_of_network):
                    if graph.number_of_nodes() > 0 and node in graph:
                        values_list = [str(i), str(node)]
                        values_list += list(graph.nodes[node].values())[:(len(attributes_list)-2)]
                        csv_writer.writerow(values_list)
            network_file_opened.close()  # Close opened fiel
        else:
            print("Nodes list is empty!")
        pass

    def write_temp_network_nodes_info_csv(
        self,
        graphs_of_network: list[nx.Graph],
        nodes: list[tuple],
    ):
        if len(nodes) > 0:
            file_existence = os.path.isfile(self.path + self.network_name + " graph temp" + ".csv")
            network_file_opened = open(
            str(self.path + self.network_name + " graph temp" + ".csv"), "a", newline="")
            csv_writer = csv.writer(network_file_opened)
            attributes_list = ["layer_id", "node_id"]
            node = nodes[0][1]
            temp_graph = None
            for graph in graphs_of_network:
                if graph.number_of_nodes() > 0:
                    if node in graph:
                        temp_graph = graph
                        break
            attributes_list += list(temp_graph.nodes[node].keys())
            del temp_graph
            if file_existence == False:
                csv_writer.writerow(attributes_list)
            for layer, node in nodes:
                    values_list = [str(layer), str(node)]
                    values_list += list(graphs_of_network[layer].nodes[node].values())[:(len(attributes_list)-2)]
                    csv_writer.writerow(values_list)
            network_file_opened.close()  # Close opened fiel
        else:
            print("Nodes list is empty!")
        pass
    
    def write_network_nodes_info_with_layer_info_csv(
            self,
            node_fraction: float,
            graphs_of_network: list[nx.Graph],
            nodes: list = None,
        ):
            if nodes is None:
                # Crete or edite network infos file
                network_file_opened = open(
                    str(self.path + self.network_name + "_fn=" + str(node_fraction) + " graph" + ".csv"), "w", newline="")
                attributes_list = ["layer_id", "node_id"]
                node = None
                temp_graph = None
                for g in graphs_of_network:
                    if g.number_of_nodes() > 0:
                        node = list(g.nodes)[0]
                        temp_graph = g
                        break
                attributes_list += list(temp_graph.nodes[node].keys())
                del temp_graph
                csv_writer = csv.writer(network_file_opened)
                csv_writer.writerow(attributes_list)
                for i, graph in enumerate(graphs_of_network):
                    for node in graph:
                        if graph.number_of_nodes() > 0:
                            values_list = [str(i), str(node)]
                            values_list += list(graph.nodes[node].values())
                            csv_writer.writerow(values_list)
                network_file_opened.close()
            elif len(nodes) > 0:
                # Crete or edite network infos file
                network_file_opened = open(
                    str(self.path + self.network_name + "_fn=" + str(node_fraction) + " graph" + ".csv"), "w", newline="")
                attributes_list = ["layer_id", "node_id"]
                node = nodes[0]
                temp_graph = None
                for g in graphs_of_network:
                    if g.number_of_nodes() > 0:
                        if node in g:
                            if node in g:
                                temp_graph = g
                                break
                attributes_list += list(temp_graph.nodes[node].keys())
                del temp_graph
                csv_writer = csv.writer(network_file_opened)
                csv_writer.writerow(attributes_list)
                for node in nodes:
                    for i, graph in enumerate(graphs_of_network):
                        if graph.number_of_nodes() > 0 and node in graph:
                            values_list = [str(i), str(node)]
                            values_list += list(graph.nodes[node].values())
                            csv_writer.writerow(values_list)
                network_file_opened.close()  # Close opened fiel
            else:
                print("Nodes list is empty!")
            pass

    def write_network_layer_infos_csv(self, graphs_of_network:list[nx.Graph], unic_str:str=''):
        network_file_opened = open(
            str(self.path + self.network_name + " layers" + unic_str + ".csv"), "w", newline="")
        csv_writer = csv.writer(network_file_opened)
        header_titles = []
        temp_graph = None
        for g in graphs_of_network:
            if g.number_of_nodes() > 0:
                temp_graph = g
        header_titles += list(temp_graph.graph.keys())
        del temp_graph
        csv_writer.writerow(header_titles)
        for i, graph in enumerate(graphs_of_network):
            if graph.number_of_nodes() > 0:
                layer_values = list(graph.graph.values())
                csv_writer.writerow(layer_values)
        pass

    def write_network_edges_infos_csv(self, graphs_of_network:list[nx.Graph]):
        # Crete or edite network infos file
        network_file_opened = open(
            str(self.path + self.network_name + " edges" + ".csv"), "w", newline=""
        )
        attributes_list = ["layer_id", "edge", "source", "source_degree", "destination", "destination_degree"]
        edge = None
        temp_graph = None
        for g in graphs_of_network:
            if g.number_of_edges() > 0:
                edge = list(g.edges)[0]
                temp_graph = g
        attributes_list += list(temp_graph.edges[edge].keys())
        attributes_list.append("in_layers_id")
        del temp_graph
        csv_writer = csv.writer(network_file_opened)
        csv_writer.writerow(attributes_list)
        for i, graph in enumerate(graphs_of_network):
            if graph.number_of_edges() > 0:
                for edge in graph.edges:
                    values_list = [str(i), str(edge), str(edge[0]),
                                    str(graph.degree(edge[0])), str(edge[1]),
                                        str(graph.degree(edge[1]))]
                    values_list += list(graph.get_edge_data(*edge).values())
                    values_list.append([])
                    for j, temp_graph in enumerate(graphs_of_network):
                        if temp_graph.number_of_nodes() > 0 and temp_graph.has_edge(*edge):
                            values_list[-1].append(j)
                    csv_writer.writerow(values_list)
        pass

    def write_node_SIR(self,
            beta: "float", landa: "float",
            epoch: "int",layer:int, node:str, sir_value:float):
        # Crete or edite network infos file
        new_file_name = (self.network_name +
                            "_beta=" + str(beta) +
                            "_landa=" + str(landa) +
                            "_epoch=" + str(epoch))
        check_file_exist = os.path.isfile(str(self.path + new_file_name + " temp SIR.csv"))
        # if check_file_exist == False:
        #     csv_file = open(str(self.path + new_file_name + " temp SIR.csv"), "a+", newline="")
        #     csv_file.close()
        csv_file = open(str(self.path + new_file_name + " temp SIR.csv"), "a", newline="")
        csv_writer = csv.writer(csv_file)
        if check_file_exist == False:
            csv_writer.writerow(['layer','node','SIR'])
        csv_writer.writerow([layer, node, sir_value])
        csv_file.close()
        pass

    def write_layer_nodes_att_in_csv(self, graph:nx.Graph, node_att:str, nodes:list[str|int]=None):
        new_path = file_handler_obj.make_dir(self.network_path + self.network_name + '/', node_att)
        file_infos = f"{new_path + str(graph.graph['id'])}.csv"
        if nodes is None:
            nodes_att = nx.get_node_attributes(graph, node_att)
            df = pd.DataFrame.from_dict(nodes_att, orient="index")
            df.to_csv(file_infos, header=False)
        elif len(nodes) > 0:
            try:
                f = open (file_infos, 'a', newline='')
                wtr = csv.writer(f)
                for node in nodes:
                    if node in graph:
                        wtr.writerow([node, graph.nodes[node][node_att]])
                f.close()
            except :
                f = open (file_infos,'w', newline='')
                wtr = csv.writer(f)
                for node in nodes:
                    if node in graph:
                        wtr.writerow([node, graph.nodes[node][node_att]])
                f.close()        
        else:
            print("Nodes list is empty!")
    @staticmethod
    def write_layers_SIR_p_to_csv(path: str, file_name: str, dict_list: dict[dict]):
        data = None
        data = pd.DataFrame.from_dict(dict_list, orient="index")
        headers = list(data.head(0))
        headers.append('AVG')
        data = data.fillna(0)
        data['average'] = data.mean(numeric_only=True, axis=1)
        pd.DataFrame.to_csv(data, path + file_name + ".csv", header=headers, index_label="Node")
        return 'Done.'

    @staticmethod
    def node_SIR_p_writer(node_by_node_SIR_p_writer_status, SIR_p_file, nodes_SIR_p_value):
        if not node_by_node_SIR_p_writer_status:
            f = open (SIR_p_file, 'w', newline='')
            wtr = csv.writer(f)
            for k, v in nodes_SIR_p_value.items():
                wtr.writerow([k, v])
            f.close()
            return True
        else:
            f = open (SIR_p_file,'a', newline='')
            wtr = csv.writer(f)
            for k, v in nodes_SIR_p_value.items():
                wtr.writerow([k, v])
            f.close()
        return True

    @staticmethod
    def write_results_in_file(root_path:os.path, file_name:str, results:list|dict):
        write_file = root_path + file_name + '.json'
        with open(write_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        pass

    pass