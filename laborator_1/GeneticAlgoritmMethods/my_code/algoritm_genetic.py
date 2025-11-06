z#!/usr/bin/python

import numpy as np

class GeneticAlgorithm(object):
    """

    """

    def __init__(self, name=""):
        super().__init__(name)

    def __call__(self):
        """
        Populatia este compusa din indivizi ce au un fitnes mai slab si elita care are cel mai mare fitness.
        Indivizii sunt compusi din alele, (o alela este un numar intreg 0..GENOME_LENGTH)
        Numarul de alele este GENOME_LENGTH + 1
        Numarul populatiei totale este 'POPULATION_SIZE', numarul elitei este 'ELITE_SIZE'.
        Indivizii care alcatuiesc elita sunt pusi in coada populatiei, pentru a face posibil ca unii indivizi din elita sa se incruciseze si cu indivizi din populatia obisnuita.
        Indivizii care fac parte din elita pot avea un numar mai mare de parteneri, dar un numar mic de copii pentru a evita cazuri de minim local.
        Indivizii din populatia simpla au numar mai mic de parteneri dar un numar mai mare de copii, pentru a diversifica populatia.
        map_distances - distanta dintre orase
        population    - populatia lista de indivizi
        """
        raise NameError("Nu este implementat corpul agoritmului genetic")

    def evolutionMonitor(self, best_distance):
        """
        Monitorizarea evolutiei de invatare: datele sunt pastrate intr-un vector
        best_distance - cea mai buna distanta
        """
        raise NameError("Nu este implementata 'evolutionMonitor'")

    def setElites(self, population, elites):
        raise NameError("Nu este implementata 'setElites'")

    def clcMetrics(self, population, fitness_values):
        """
        Calculare metrici:
            population - populatia compusa dintr-o lista de indivizi
            fitness_values - valorile fitnes pentru fiecare individ
        """
        raise NameError("Nu este implementata calcularea metricilor")

    def showMetrics(self, generation, d_info):
        """Afisare metrici"""
        metric_info ="Name:{}, Generatia: {}, ".format(self.__name, generation)
        for key in d_info.keys():
            val = d_info[key]
            if (isinstance(val, float)):
                val = round(val, 3)
            metric_info + ="{}: {},".format(key, val)
        print(metric_info)

    def findSimilarIndivids(self, population, individ, tolerance):
        """
        Cauta indivizi din intreaga populatie ce are codul genetic identic cu un individ,
        population - lista de indivizi
        individ    - vector compus din codul genetic
        tolerance  - cate gene pot fi diferite
        """
        tmp = (population==individ).sum(axis=1)
        return np.argwhere(tmp>=tolerance)

    def similarIndivids(self, population):
        """
        Returneaza un vector de flaguri pentru fiecare individ din populatie daca este gasit codul genetic si la alti indivizi
        population - lista de indivizi
        """
        # initializare vector de flaguri pentru fiecare individ
        similar_args_flag = np.zeros(self.POPULATION_SIZE, dtype=bool)
        # setare toleranta, numarul total de gene
        tolerance = self.GENOME_LENGTH
        # 
        for i in range(self.POPULATION_SIZE-1, -1, -1):
            if (similar_args_flag[i]):
                pass
            else:
                individ = population[i]
                similar_args = self.findSimilarIndivids(population, individ, tolerance)
                #print("similar_args", similar_args)
                similar_args_flag[similar_args] = True
                similar_args_flag[i] = False # scoate flagul de pe individul care este copiat
        return similar_args_flag

    def permuteSimilarIndivids(self, population):
        """
        Returneaza un vector de flaguri pentru fiecare individ din populatie daca este gasit codul genetic si la alti indivizi
        population - lista de indivizi
        """
        raise NameError("Nu este implementata functia 'permuteSimilarIndivids'!!!")

    def stres(self, population, fitness_values, best_individ, best_distance):
        """Aplica stres asupra populatiei.
        Functia de stres, se aplica atunci cand ajungem intr-un minim local,
        cauta cele mai frecvente secvente de genom si aplica un stres modifica acele zone
        population    - populatia
        best_individ  - individul cu cel mai bun fitness
        best_distance - cea mai buna distanta
        """
        raise NameError("Nu este implementata functia 'permuteSimilarIndivids'!!!")

    def getArgBest(self, fitness_values):
        """Cautarea rutei optime din populatie"""
        index = np.argmax(fitness_values, axis=None, keepdims=False)
        return index

    def getArgBestChild(self, fitness_values, size):
        """Returneaza pozitiile 'size' cu cele mai mari valori, ale fitnesului
        fitness_values - valorile fitness pentru un copil
        size           - numarul de argumente cu cei mai buni copii
        """
        args = np.argpartition(fitness_values,-size)
        return args[-size:]

    def getArgWeaks(self, fitness_values, size):
        """Returneaza pozitiile 'size' cu cele mai mici valori, ale fitnesului
        fitness_values - valorile fitness a populatiei
        size           - numarul de argumente cu cei mai buni indivizi
        """
        args = np.argpartition(fitness_values, size)
        return args[:size]

    def getArgElite(self, fitness_values):
        """Returneaza pozitiile 'ELITE_SIZE' cu cele mai mari valori, ale fitnesului
        fitness_values - valorile fitness a populatiei
        """
        args = np.argpartition(fitness_values,-self.ELITE_SIZE)
        args = args[-self.ELITE_SIZE:]
        return args
