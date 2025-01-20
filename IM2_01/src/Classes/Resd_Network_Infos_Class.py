# version 1.0.1
import os
import numpy as np


class Resd_Network_Infos:

    @staticmethod
    def read_nodeFrom_layerFrom_nodeTo_layerTo(
        network_path: str,
        network_name: str,
        network_type: str,
        directed: bool = False,
        weighted: bool = False,
    ):
        network_layers_info = []
        entra_layer_edges = []
        inter_layer_edge = {}
        network_layers_nodes = []
        entra_layer_edges_features = {}
        layer_id = 0
        try:
            file_opened = open(network_path + network_name + network_type)
            content = file_opened.readlines()
            file_opened.close()
            i = 0
            for item in content:
                if item != None and item.strip() != "" and item.strip()[0] != "#":
                    content_parts = item.strip().split(" ")
                    if len(content_parts) > 3:
                        source_node_id = content_parts[0].strip()
                        source_node_layer_id = content_parts[1].strip()
                        destination_node_id = content_parts[2].strip()
                        destination_node_layer_id = content_parts[3].strip()

                        layer_id = int(source_node_layer_id)
                        while len(network_layers_nodes) < layer_id + 1:
                            network_layers_info.append([])
                            entra_layer_edges.append([])
                            network_layers_nodes.append([])

                        layer_id = int(destination_node_layer_id)
                        while len(network_layers_nodes) < layer_id + 1:
                            network_layers_info.append([])
                            entra_layer_edges.append([])
                            network_layers_nodes.append([])

                        edge_info = list()
                        edge_info.append(source_node_id)
                        edge_info.append(destination_node_id)
                        edge_features = []
                        if len(content_parts) > 4:
                            j = 4
                            while j < len(content_parts):
                                edge_features.append(content_parts[j])
                                j += 1

                        if source_node_layer_id == destination_node_layer_id:
                            layer_id = int(source_node_layer_id)
                            layer_info = list()
                            layer_info.append(layer_id)
                            layer_info.append(str(layer_id))

                            network_layers_info[layer_id] = layer_info
                            network_layers_nodes[layer_id].append(source_node_id)
                            network_layers_nodes[layer_id].append(destination_node_id)
                            entra_layer_edges[layer_id].append(edge_info)
                            if weighted:
                                edge_key = str(
                                    source_node_layer_id + " " + source_node_id + " " + destination_node_id
                                )
                                entra_layer_edges_features[edge_key] = edge_features

                        else:
                            if not inter_layer_edge.get(source_node_layer_id):
                                inter_layer_edge[source_node_layer_id] = {}
                            if not inter_layer_edge[source_node_layer_id].get(source_node_id):
                                inter_layer_edge[source_node_layer_id][source_node_id] = {}
                            if not inter_layer_edge[source_node_layer_id][source_node_id].get(destination_node_layer_id):
                                if weighted:
                                    inter_layer_edge[source_node_layer_id][source_node_id][destination_node_layer_id] = {}
                                    if not inter_layer_edge[source_node_layer_id][source_node_id][destination_node_layer_id].get(destination_node_id):
                                        inter_layer_edge[source_node_layer_id][source_node_id][destination_node_layer_id][destination_node_id] = {}
                                    inter_layer_edge[source_node_layer_id][source_node_id][destination_node_layer_id][destination_node_id] = edge_features
                                else:
                                    inter_layer_edge[source_node_layer_id][source_node_id][destination_node_layer_id] = []
                                    inter_layer_edge[source_node_layer_id][source_node_id][destination_node_layer_id].append(destination_node_id)
                            if not directed:
                                if not inter_layer_edge.get(destination_node_layer_id):
                                    inter_layer_edge[destination_node_layer_id] = {}
                                if not inter_layer_edge[destination_node_layer_id].get(destination_node_id):
                                    inter_layer_edge[destination_node_layer_id][destination_node_id] = {}
                                if not inter_layer_edge[destination_node_layer_id][
                                    destination_node_id].get(source_node_layer_id):
                                    if weighted:
                                        inter_layer_edge[destination_node_layer_id][destination_node_id][source_node_layer_id] = {}
                                        if not inter_layer_edge[destination_node_layer_id][destination_node_id][source_node_layer_id].get(source_node_id):
                                            inter_layer_edge[destination_node_layer_id][destination_node_id][source_node_layer_id][source_node_id] = {}
                                        inter_layer_edge[destination_node_layer_id][destination_node_id][source_node_layer_id][source_node_id] = edge_features
                                    else:
                                        inter_layer_edge[destination_node_layer_id][destination_node_id][source_node_layer_id] = []
                                        inter_layer_edge[destination_node_layer_id][destination_node_id][source_node_layer_id].append(source_node_id)

            # file_opened.close()
        except Exception as e:
            print(e)

        return (
            network_layers_info,
            network_layers_nodes,
            entra_layer_edges,
            entra_layer_edges_features,
            inter_layer_edge,
        )

    @staticmethod
    def read_layer_node_node_weight_dataset(file_path: os.path) -> dict:
        inputDataset = np.loadtxt(file_path)
        inputDataset = np.delete(inputDataset, [-1], axis=1)
        edgeList = {}
        for i in inputDataset:
            edgeList.setdefault(i[0], []).append((int(i[1]), int(i[2])))
        edgeList = edgeList.values()
        return edgeList
    pass