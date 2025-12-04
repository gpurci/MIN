#!/usr/bin/python

import numpy as np

def neighbors_parents(parent1, parent2):
    # gasseste vecinii
    tmp1 = np.roll(parent1,  1).reshape(1, -1)
    tmp2 = np.roll(parent1, -1).reshape(1, -1)
    neighbors1 = np.concatenate((tmp1, tmp2), axis=0)
    tmp1 = np.roll(parent2,  1).reshape(1, -1)
    tmp2 = np.roll(parent2, -1).reshape(1, -1)
    neighbors2 = np.concatenate((tmp1, tmp2), axis=0)
    # sortarea pozitiei oraselor
    sort_pos_p1 = np.argsort(parent1)
    sort_pos_p2 = np.argsort(parent2)
    #
    unique_tmp = [] # gasirea vecinilor unui oras pentru ambii parinti
    neighbors1 = neighbors1.transpose()[sort_pos_p1]
    neighbors2 = neighbors2.transpose()[sort_pos_p2]
    for col1, col2 in zip(neighbors1, neighbors2):
        unique = np.union1d(col1, col2)
        unique_tmp.append(unique)
    return unique_tmp

def erx_crossover(parent1, parent2):
    # mosteneste parinte1
    parents_neighbors = neighbors_parents(parent1, parent2) # sorted
    visited_city = np.zeros(parent1.shape[0], dtype=bool)
    offspring    = parent1.copy()
    GENOME_LENGTH = parent1.shape[0]
    idx = 0
    while (idx < (GENOME_LENGTH-1)):
        # selectam orasul de start
        start_city = offspring[idx]
        visited_city[start_city] = True
        # obtinem toti vecinii pentru orasul actual
        city_neighbors = parents_neighbors[start_city]
        # setarea by default a urmatorului vecin
        sel_neighbor   = offspring[idx+1]
        # setam numarul minim de vecini
        min_neighbors  = GENOME_LENGTH
        # gasim orasul cu cel mai mic numar de vecini, nevizitati
        for neighbor in city_neighbors[::-1]:
            # numarul de vecini
            act_size = parents_neighbors[neighbor].shape[0]
            # 
            if ((min_neighbors > act_size) and (visited_city[neighbor] == False)):
                min_neighbors = act_size
                sel_neighbor  = neighbor
        else: # setam orasul vizitat
            offspring[idx+1] = sel_neighbor
        # mergem catre urmatorul oras
        idx += 1
    # returneaza mostenitorul
    return offspring

