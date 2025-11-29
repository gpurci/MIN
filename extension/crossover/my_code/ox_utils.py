#!/usr/bin/python

import numpy as np

def ox_crossover_order_parent2(parent1, parent2, locus):
    # mosteneste parinte1
    offspring = parent1.copy()
    # obtinerea genelor de pe locus parent1
    genes_p1    = parent1[locus]
    # gasirea locusurilor genelor din parent 1 in parent2 
    _, pos_p2   = np.nonzero(parent2 == genes_p1.reshape(-1, 1))
    # aranjarea genelor dupa ordinea din parent 2
    sort_pos_p2 = np.argsort(pos_p2)
    # salvarea genelor dupa ordinea din parent 2
    offspring[locus] = genes_p1[sort_pos_p2]
    return offspring

def find_similar_zones(mask_genes, start, lenght, arg):
    """Cautarea celei mai mari zone, in care genele sunt identice,
    sau cauta cea mai mare secveta de unitati"""
    if (arg < mask_genes.shape[0]):
        tmp_arg = arg
        tmp_st  = arg
        tmp_lenght = 0
        while tmp_arg < mask_genes.shape[0]:
            if (mask_genes[tmp_arg]):
                tmp_arg   += 1
            else:
                tmp_lenght = tmp_arg - tmp_st
                if (lenght < tmp_lenght):
                    start, lenght = tmp_st, tmp_lenght
                return find_similar_zones(mask_genes, start, lenght, tmp_arg+1)
    else:
        return start, lenght
    return start, lenght

def sim_inversion_field(parent1, parent2, start, subset_size, genome_length):
    # check diff
    mask_sim_locus = parent1==parent2
    if (mask_sim_locus.sum() == mask_sim_locus.shape[0]):
        locus1 = max(0, start-subset_size//3)
        locus2 = min(genome_length, start+subset_size//3)
        parent2[locus1:locus2] = np.flip(parent2[locus1:locus2])
    return parent2

def sim_scramble_field(parent1, parent2, start, subset_size, genome_length):
    # check diff
    mask_sim_locus = parent1==parent2
    if (mask_sim_locus.sum() == mask_sim_locus.shape[0]):
        locus1 = max(0, start-subset_size//3)
        locus2 = min(genome_length, start+subset_size//3)
        parent2[locus1:locus2] = np.random.permutation(parent2[locus1:locus2])
    return parent2

def sim_shift_field(parent1, parent2, start, subset_size, genome_length):
    # check diff
    mask_sim_locus = parent1==parent2
    if (mask_sim_locus.sum() == mask_sim_locus.shape[0]):
        locus1 = max(0, start-subset_size//3)
        locus2 = min(genome_length, start+subset_size//3)
        size_shift = np.random.randint(low=1, high=subset_size, size=None)
        parent2[locus1:locus2] = np.roll(parent2[locus1:locus2], size_shift)
    return parent2

