import networkx as nx
import random
import math

class Memetic():

    @staticmethod
    def get_sigma_hat_S(graph:nx.Graph, S:list[str|int], p:float=0.01):
        sigma_hat_s = len(S)
        for s in S:
            cs = list(set(graph.neighbors(s)) - set(S))
            temp_1 = 0
            for c in cs:
                p_s_c = (p * (graph.degree(s) / graph.degree(c)))
                temp_2 = 1
                cc = list(set(graph.neighbors(c)) - set(S))
                for d in cc:
                    p_c_d = (p * (graph.degree(c) / graph.degree(d)))
                    temp_2 += p_c_d
                temp_1 += p_s_c * temp_2
            sigma_hat_s += temp_1
        return sigma_hat_s

    @staticmethod
    def get_n_hop_nbrs(graph:nx.Graph, node:str|int, n_hop:int):
        if node in graph:
            nbrs = []
            if n_hop <= 0:
                return nbrs
            else:
                nbrs.extend(list(graph.neighbors(node)))
                new_nbrs = nbrs.copy()
            c = 2
            while c <= n_hop:
                new_nbrs_len = len(new_nbrs)
                for nbr in range(new_nbrs_len):
                    new_nbrs.extend(list(graph.neighbors(new_nbrs[nbr])))
                new_nbrs = list(set(new_nbrs) - set(nbrs))
                nbrs.extend(new_nbrs)
                c += 1
            return list(set(nbrs))
        return None

    @staticmethod
    def overlapping_influence_chi(seed_set:list[str|int], graph:nx.Graph, p:float=0.01):
        chi = 0
        for s in seed_set:
            if graph.number_of_edges() > 0 and s in graph:
                cs = list(graph.neighbors(s))
                cs = list(set(cs) - set(seed_set))
                for c in cs:
                    intersection_hs_s = set.intersection(set(cs), set(set(seed_set) - set(s)))
                    for d in intersection_hs_s:
                        p_s_c = (p * (graph.degree(s) / graph.degree(c)))
                        p_c_d = (p * (graph.degree(c) / graph.degree(d)))
                        chi += (p_s_c * p_c_d)
        return chi
    
    @staticmethod
    def get_sigma_1_c(c:str|int, graph:nx.Graph, p:float=0.01):
        sigma_1_c = 1
        if graph.number_of_edges() > 0 and c in graph:
            nbrs = list(graph.neighbors(c))
            for nbr in nbrs:
                sigma_1_c += (p * (graph.degree(c) / graph.degree(nbr)))
        return sigma_1_c

    @staticmethod
    def spreading_approximation(S:list[str|int], graph:nx.Graph, p:float=0.01):
        sigma_hat_S = 0
        chi = Memetic.overlapping_influence_chi(S, graph, p) # Denotes the overlapping influence from active nodes to the original seeds
        for s in S:
            if graph.number_of_edges() > 0 and s in graph:
                sigma_hat_s = Memetic.get_sigma_hat_S(graph, [s], p)
                temp_1 = 0
                for s1 in S:
                    cs = list(set.intersection(set(graph.neighbors(s)), set(S)))
                    temp_2 = 0
                    for c in cs:
                        p_s_c = (p * (graph.degree(s) / graph.degree(c)))
                        sigma_1_c = Memetic.get_sigma_1_c(c, graph, p)
                        p_c_s = (p * (graph.degree(c) / graph.degree(s)))
                        temp_2 += (p_s_c * (sigma_1_c - p_c_s))
                    temp_1 += temp_2
                sigma_hat_s -= temp_1 - chi
                sigma_hat_S += sigma_hat_s
        return sigma_hat_S

    @staticmethod
    def spreading_approximation_multilayer(S:list[str|int], graphs_of_network:list[nx.Graph], p:float=0.01):
        multilayer_spread = 0
        for graph in graphs_of_network:
            if graph.number_of_nodes() > 0:
                multilayer_spread += Memetic.spreading_approximation(S,graph,p)
        return multilayer_spread

    @staticmethod
    def crossover_indicators(k:int):
        validate_flag = False
        if k > 2:
            while validate_flag == False:
                p1 = random.randint(0, k-1)
                p2 = random.randint(0, k-1)
                if p2 < p1:
                    temp = p1
                    p1 = p2
                    p2 = temp
                if (p2 - p1) > 0 and (p2 - p1) < (k-1):
                    validate_flag = True
            return p1, p2
        elif k == 2:
            return 1, 2
        else:
            return 1, 1

    @staticmethod
    def gen_validator(gen:list, entier_population:list[tuple]):
        temp_gen = gen.copy()
        duplicates_indexes = []
        i = 0
        while i < len(temp_gen):
            j = i + 1
            while j < len(temp_gen):
                if temp_gen[i] == temp_gen[j]:
                    duplicates_indexes.append(j)
                j += 1
            i += 1
        duplicates_indexes = list(set(duplicates_indexes))
        if len(duplicates_indexes) > 0:
            for item in duplicates_indexes:
                while True:
                    random_sample =random.choice(entier_population)[0]
                    if not(random_sample in temp_gen):
                        temp_gen[item] = random_sample
                        break
            return temp_gen
        else:
            return gen

    @staticmethod
    def crossover(gen_a:list, gen_b:list, entier_population:list[tuple]):
        new_gen_a = gen_a.copy()
        new_gen_b = gen_b.copy()
        p1, p2 = Memetic.crossover_indicators(len(gen_a))
        a = new_gen_a[p1:p2]
        b = new_gen_b[p1:p2]
        new_gen_a[p1:p2] = b
        new_gen_b[p1:p2] = a        
        new_gen_b = Memetic.gen_validator(new_gen_b, entier_population)
        new_gen_a = Memetic.gen_validator(new_gen_a, entier_population)
        
        return new_gen_a, new_gen_b
        
    @staticmethod
    def new_candid_selecor(gen:list, p1:int, snen_by_synthetic_degree:list[tuple], graphs_of_network:list[nx.Graph]):
        new_candids = random.sample(snen_by_synthetic_degree, len(gen))
        selected_candid_score = -1
        selected_candid = None
        for candid, degree in new_candids:
            dist = Memetic.multilayer_distance_n_o(graphs_of_network, len(snen_by_synthetic_degree),gen[p1], candid)
            score = degree * dist
            if score > selected_candid_score:
                selected_candid_score = score
                selected_candid = candid
        return selected_candid
        
    @staticmethod
    def mutation(gen:list, snen_by_synthetic_degree:list[tuple], graphs_of_network:list[nx.Graph], k:int):
        new_gen = gen.copy()
        p1 = random.randint(0, k-1)
        validate_flag = False
        while validate_flag == False:
            selected_candid = Memetic.new_candid_selecor(new_gen, p1, snen_by_synthetic_degree, graphs_of_network)
            if not(selected_candid in new_gen):
                new_gen[p1] = selected_candid
                validate_flag = True
        return new_gen
    
    @staticmethod
    def get_max_degree_neighbor(node:str|int, graphs_of_network:list[nx.Graph]):
        final_max_degree_nbr = (None, None)
        for graph in graphs_of_network:
            if graph.number_of_edges() > 0 and node in graph:
                nbrs = list(graph.neighbors(node))
                nbrs_degree = dict(graph.degree(nbrs))
                max_degree_nbr = max(nbrs_degree, key=nbrs_degree.get)
                if final_max_degree_nbr[0] is None:
                    final_max_degree_nbr =(max_degree_nbr, nbrs_degree[max_degree_nbr])
                else:
                    if final_max_degree_nbr[1] < nbrs_degree[max_degree_nbr]:
                        final_max_degree_nbr =(max_degree_nbr, nbrs_degree[max_degree_nbr])
        return final_max_degree_nbr[0]

    @staticmethod
    def get_max_degree_neighbor_in_layer(node:str|int, graph:nx.Graph):
        max_degree_nbr = None
        if graph.number_of_edges() > 0 and node in graph:
            nbrs = list(graph.neighbors(node))
            nbrs_degree = dict(graph.degree(nbrs))
            max_degree_nbr = max(nbrs_degree, key=nbrs_degree.get)
        return max_degree_nbr

    @staticmethod
    def get_min_synthetic_degree_seed(gen:list, snen_by_synthetic_degree:dict):
        new_gen = gen.copy()
        min_synthetic_degree_seed = (None, None)
        for seed in new_gen:
            synthetic_degree_value = snen_by_synthetic_degree[seed]
            if min_synthetic_degree_seed[0] is None:
                min_synthetic_degree_seed = (seed, synthetic_degree_value)
            else:
                if synthetic_degree_value < min_synthetic_degree_seed[1]:
                    min_synthetic_degree_seed = (seed, synthetic_degree_value)
        return min_synthetic_degree_seed[0]

    @staticmethod
    def get_randome_validate_layer(graphs_of_network:list[nx.Graph]):
        while True:
            p = random.randint(0, len(graphs_of_network) - 1)
            if graphs_of_network[p].number_of_edges() > 0 :
                return graphs_of_network[p]

    @staticmethod
    def get_max_degree_node_in_graph(graph:nx.Graph):
        degree_list = dict(graph.degree())
        return max(degree_list, key=degree_list.get)

    @staticmethod
    def local_search_1(gen:list, entier_population:list[str|int], graphs_of_network:list[nx.Graph], p:float=0.01):
        new_gen = gen.copy()
        gen_spread_approximation = -1
        max_spread_gen = None
        temp_gen = gen.copy()
        for i, item in enumerate(new_gen):
            for graph in graphs_of_network:
                if graph.number_of_edges() > 0 and item in graph:
                    max_degree_nbr = Memetic.get_max_degree_neighbor_in_layer(item, graph)
                    if max_degree_nbr in entier_population:
                        temp_gen[i] = max_degree_nbr
                    temp_gen = Memetic.gen_validator(temp_gen, entier_population)
                    temp_gen_spread_approximation = Memetic.spreading_approximation(temp_gen, graph, p)
                    if temp_gen_spread_approximation > gen_spread_approximation:
                        gen_spread_approximation = temp_gen_spread_approximation
                        max_spread_gen = temp_gen



        return max_spread_gen

    @staticmethod
    def local_search_2(gen:list, entier_population:list[str|int], snen_by_synthetic_degree:dict, graphs_of_network:list[nx.Graph]):
        new_gen = gen.copy()
        min_synthetic_degree_seed = Memetic.get_min_synthetic_degree_seed(new_gen, snen_by_synthetic_degree)
        min_synthetic_degree_seed_index = new_gen.index(min_synthetic_degree_seed)
        selected_layer = Memetic.get_randome_validate_layer(graphs_of_network)
        new_seed = Memetic.get_max_degree_node_in_graph(selected_layer)
        new_gen[min_synthetic_degree_seed_index] = new_seed
        new_gen = Memetic.gen_validator(new_gen, entier_population)

        return new_gen
    
    @staticmethod
    def str_to_int_func(input_str:str):
        if input_str.isdigit():
            return int(input_str)
        pass

    def n_hop_nbrs(graph, node, n_hop):
        if node in graph:
            nbrs = []
            if n_hop <= 0:
                return nbrs
            else:
                nbrs.extend(list(graph.neighbors(node)))
                new_nbrs = nbrs.copy()
            c = 2
            while c <= n_hop:
                new_nbrs_len = len(new_nbrs)
                for nbr in range(new_nbrs_len):
                    new_nbrs.extend(list(graph.neighbors(new_nbrs[nbr])))
                new_nbrs = list(set(new_nbrs) - set(nbrs))
                nbrs.extend(new_nbrs)
                c += 1
            return list(set(nbrs))
        return None
    
    @staticmethod
    def distance_n_o(graph, N, n, o):
        nbs_n = Memetic.n_hop_nbrs(graph, n, 2)
        nbs_o =Memetic.n_hop_nbrs(graph, o, 2)
        dist = len(set(nbs_n).symmetric_difference(nbs_o)) / N
        return dist
    
    @staticmethod
    def multilayer_distance_n_o(graphs_of_network, N, n, o):
        nbs_n = []
        nbs_o = []
        for graph in graphs_of_network:
            if graph.number_of_edges() > 0:
                if n in graph and o in graph:
                    nbs_n.extend(Memetic.n_hop_nbrs(graph, n, 2))
                    nbs_o.extend(Memetic.n_hop_nbrs(graph, o, 2))
        dist = (len(set(nbs_n).symmetric_difference(set(nbs_o)))) / N
        return dist

    @staticmethod
    def sort_networks_entier_nodes_by_synthetic_degree(graphs_of_network, network_entier_nodes_list, k):
        nodes_synthetic_degree = []
        synthetic_degree_mean = 0
        for node in network_entier_nodes_list:
            synthetic_degree = 0
            for j, graph in enumerate(graphs_of_network):
                if graph.number_of_edges() > 0 and node in graph:
                    synthetic_degree += graph.degree(node)
            nodes_synthetic_degree.append((node, synthetic_degree))
            synthetic_degree_mean += synthetic_degree
            
        nodes_synthetic_degree = sorted(nodes_synthetic_degree, key=lambda a: a[1], reverse=True)
        synthetic_degree_mean = (synthetic_degree_mean / len(network_entier_nodes_list))
        quarter_top = math.ceil(len(network_entier_nodes_list) / k)
        k_top_index = quarter_top
        i = quarter_top
        while i < len(nodes_synthetic_degree):
            if nodes_synthetic_degree[quarter_top][1] > nodes_synthetic_degree[i][1]:
                k_top_index = i - 1
                break
            i += 1
        return nodes_synthetic_degree, synthetic_degree_mean, k_top_index

    @staticmethod
    def select_k_nodes_by_synthetic_degree(k_top_synthetic_degree_list, bottom_synthetic_degree_list, k):
        seed = []
        i = 0
        while i < k :
            if len(seed) >= k:
                    break
            j = 0
            while j < 2:
                random_sample = random.choice(k_top_synthetic_degree_list)
                if not(random_sample in seed):
                    seed.append(random_sample)
                    j += 1
                if len(seed) >= k:
                    break
            if len(seed) >= k:
                    break
            j = 0
            while j < 1:
                random_sample = random.choice(bottom_synthetic_degree_list)
                if not(random_sample in seed):
                    seed.append(random_sample)
                    j += 1
                if len(seed) >= k:
                    break
            if len(seed) >= k:
                    break
            i += 3
        for j, node in enumerate(seed):
            seed[j] = node[0]
        return seed

    @staticmethod
    def select_k_nodes_by_highest_degree(k_top_synthetic_degree_list, bottom_synthetic_degree_list, 
                                        graphs_of_network, network_entier_nodes_list, k):
        p_seed = [random.choice(k_top_synthetic_degree_list)[0]]
        i = 0
        dist_arr = []
        while i < k:
            if p_seed[0] != k_top_synthetic_degree_list[i][0]:
                dist_arr.append((k_top_synthetic_degree_list[i][0], 
                                Memetic.multilayer_distance_n_o(graphs_of_network, len(network_entier_nodes_list), p_seed[0],
                                                        k_top_synthetic_degree_list[i][0])))
            i += 1
        if len(dist_arr) > 0:
            max_dist_node = max(dist_arr, key=lambda a: a[1])[0]
            p_seed.append(max_dist_node)
            i = 2
            while i < k:
                random_sample = random.choice(network_entier_nodes_list)
                if not(random_sample in p_seed):
                    p_seed.append(random_sample)
                    i += 1

        return p_seed

