#!/usr/bin/python

import numpy as np

def positiveInsertion(offspring, locus1, locus2):
    gene    = offspring[locus1]
    # make change locuses
    locuses = np.arange(locus2+1, locus1, dtype=np.int32)
    offspring[locuses+1] = offspring[locuses]
    offspring[locus2+1]  = gene
    return offspring

def negativeInsertion(offspring, locus1, locus2):
    gene    = offspring[locus1]
    # make change locuses
    locuses = np.arange(locus1, locus2, dtype=np.int32)
    offspring[locuses] = offspring[locuses+1]
    offspring[locus2]  = gene
    return offspring

def insertion(offspring, locus1, locus2):
    if (locus1 > locus2):
        offspring = positiveInsertion(offspring, locus1, locus2)
    else:
        offspring = negativeInsertion(offspring, locus1, locus2)
    return offspring