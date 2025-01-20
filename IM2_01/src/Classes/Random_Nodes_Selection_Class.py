import math
import random
import sys
import os
from easygui import *

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if not (CURRENT_DIR in sys.path):
    sys.path.append(CURRENT_DIR)

from Files_Handler_Class import Files_Handler
file_handler_obj = Files_Handler

class Random_Nodes_Selection:

    @staticmethod
    def get_node_fraction(network_size):
        if network_size >= 1024000:
            return 0.02
        elif network_size >= 512000:
            return 0.04
        elif network_size >= 256000:
            return 0.06
        elif network_size >= 128000:
            return 0.08
        elif network_size >= 64000:
            return 0.1
        elif network_size >= 32000:
            return 0.2
        elif network_size >= 16000:
            return 0.4
        elif network_size >= 8000:
            return 0.6
        elif network_size >= 4000:
            return 0.8
        else:
            return 1

    @staticmethod
    def get_random_nodes_selection(choice:str, choices:list[str], population:list[list], degree_mean:float, node_fraction:float):
        random_selected_nodes = []
        n = len(population)
        i = len(population) - 1
        lower_bound = (degree_mean - (math.sqrt(degree_mean)))
        upper_bound = (degree_mean + (math.sqrt(degree_mean)))
        if choice == choices[1]:
            while i >= lower_bound:
                if len(population[i]) > lower_bound:
                    fraction =  math.ceil(len(population[i]) * (node_fraction))
                    if fraction > len(population[i]):
                        fraction = len(population[i])
                    random_selected_nodes.extend(random.sample(population[i], fraction))
                i -= 1
        elif choice == choices[2]:
            i = int(upper_bound)
            while i >= lower_bound:
                if len(population[i]) > 0:
                    fraction =  math.ceil(len(population[i]) * (node_fraction))
                    if fraction > len(population[i]):
                        fraction = len(population[i])
                    random_selected_nodes.extend(random.sample(population[i], fraction))
                i -= 1
        elif choice == choices[3]:
            while i >= upper_bound:
                if len(population[i]) > 0:
                    fraction =  math.ceil(len(population[i]) * (node_fraction))
                    if fraction > len(population[i]):
                        fraction = len(population[i])
                    random_selected_nodes.extend(random.sample(population[i], fraction))
                i -= 1
        else :
            while i >= 0:
                if len(population[i]) > 0:
                    if i > lower_bound:
                        fraction =  2 * (math.ceil(len(population[i]) * (node_fraction)))
                    else:
                        fraction =  math.ceil(len(population[i]) * (node_fraction))
                    if fraction > len(population[i]):
                        fraction = len(population[i])
                    random_selected_nodes.extend(random.sample(population[i], fraction))
                i -= 1
        return random_selected_nodes
