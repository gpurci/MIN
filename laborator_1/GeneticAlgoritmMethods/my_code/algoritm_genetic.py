#!/usr/bin/python

import numpy as np
from root_GA import *
from callback import *
from crossover import *
from fitness import *
from individ_repair import *
from init_population import *
from metrics import *
from mutate import *
from select_parent import *

class GeneticAlgorithm(RootGA):
    """
    Managerul de configuratie al algoritmului genetic,
    """
    def __init__(self, name="", **configs):
        super().__init__()
        self.__name = name
        self.__configs = configs
        self.__setConfig(**configs)
        self.__score_evolution = np.zeros(5, dtype=np.float32)

    def __str__(self):
        str_info = "Name: {}\n{}".format(self.__name, super().__str__())
        str_info += "\nConfigs:"
        for key in self.__configs.keys():
            str_info += "\n\t{}: {}".format(key, self.__configs[key])
        str_info += "\n"
        return str_info

    def __call__(self, population):
        """
        Manager pentru a face sincronizarea dintre toate functionalele
        """
        # initiaizarea populatiei
        if (population is None):
            population = self.initPopulation(self.POPULATION_SIZE)
        # init fitness value
        fitness_values = self.fitness(population)
        # obtinerea pozitiei pentru elite
        args_elite = self.getArgsElite(fitness_values)

        # evolutia generatiilor
        for generation in range(self.GENERATIONS):
            # nasterea unei noi generatii
            new_population = []
            # start selectie populatie
            self.selectParent1.startEpoch(fitness_values)
            self.selectParent2.startEpoch(fitness_values)
            for _ in range(self.POPULATION_SIZE):
                # selectarea positia parinte 1
                arg_parent1 = self.selectParent1()
                # selectarea positia parinte 1
                arg_parent2 = self.selectParent2()
                # obtinerea parintilor
                parent1 = population[arg_parent1]
                parent2 = population[arg_parent2]
                # incrucisarea parintilor
                offspring = self.crossover(parent1, parent2)
                # mutatii
                offspring = self.mutate(parent1, parent2, offspring) # in_place operation
                # adauga urmasii la noua generatie
                new_population.append(offspring)
            # obtinerea indivizilor ce fac parte din elita
            elite_individs = population[args_elite]
            # schimbarea generatiei
            population = np.array(new_population)
            # adaugare elita in noua populatie
            self.setElites(population, elite_individs)
            # calculare fitness
            fitness_values = self.fitness(population)
            # obtinerea pozitiei pentru elite
            args_elite     = self.getArgsElite(fitness_values)
            # calculare metrici
            scores = self.metrics.getScore(population, fitness_values)
            #self.evolutionMonitor(metrics_values)
            #self.log(population, fitness_values, args_elite, elite_individs, best_distance)
            # adaugare stres in populatie atunci cand lipseste progresul
            #fitness_values = self.stres(population, fitness_values, best_individ, best_distance)
            # afisare metrici
            self.showMetrics(generation, scores)
            # salveaza istoricul
            self.callback(generation, scores)

        return best_individ, population

    def __setConfig(self, **configs):
        # configurare metrici
        config       = configs.get("metric", None)
        self.metrics = Metrics(config)
        # configurare initializare populatie
        config       = configs.get("init_population", None)
        self.initPopulation = InitPopulation(config, self.metrics)
        # configurare fitness
        config       = configs.get("fitness", None)
        self.fitness = Fitness(config, self.metrics)
        # configurate selectie parinti
        config       = configs.get("select_parent", {"test1":0, "test2":0})
        conf_select1 = config.get("select_parent1", None)
        self.selectParent1 = SelectParent(conf_select1)
        conf_select2 = config.get("select_parent2", None)
        self.selectParent2 = SelectParent(conf_select2)
        # configurare incrucisare
        config       = configs.get("crossover", None)
        self.crossover = Crossover(config)
        # configurare mutatie
        config       = configs.get("mutate", None)
        self.mutate  = Mutate(config)
        # configurare reparatie individ, repara individ atunci cand nu este progres, o operatie de mutatie pe mai multe gene
        config       = configs.get("repair", None)
        self.individRepair = IndividRepair(config)
        # configurare callback salvare, istoricul de antrenare
        filename     = configs.get("callback", None)
        self.callback = Callback(filename)

    def help(self):
        info  = "'metric': "+self.metrics.help()
        info += "'init_population': "+self.initPopulation.help()
        info += "'fitness': "+self.fitness.help()
        info += "'select_parent': {'select_parent1': 'select_parent2'}: "+self.selectParent1.help()
        info += "'crossover': "+self.crossover.help()
        info += "'mutate': "+self.mutate.help()
        info += "'repair': "+self.individRepair.help()
        info += "'callback': "+self.callback.help()
        print(info)

    def setDataset(self, dataset):
        self.metrics.setDataset(dataset)

    def setParameters(self, **kw):
        super().setParameters(**kw)
        self.metrics.setParameters(**kw)
        self.initPopulation.setParameters(**kw)
        self.fitness.setParameters(**kw)
        self.selectParent1.setParameters(**kw)
        self.selectParent2.setParameters(**kw)
        self.crossover.setParameters(**kw)
        self.mutate.setParameters(**kw)
        self.individRepair.setParameters(**kw)

    def evolutionMonitor(self, best_score):
        """
        Monitorizarea evolutiei de invatare: datele sunt pastrate intr-un vector
        best_score - cel mai bun scor
        """
        self.__score_evolution[:-1] = self.__score_evolution[1:]
        self.__score_evolution[-1]  = best_score

    def setElites(self, population, elites):
        if (population is None):
            population = self.initPopulation(self.POPULATION_SIZE)
        fitness_values = self.fitness(population)
        args = self.getArgsWeaks(fitness_values, self.ELITE_SIZE)
        population[args] = elites
        return population

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
            metric_info +="{}: {},".format(key, val)
        print(metric_info)

    def stres(self, population, fitness_values, best_individ, best_score):
        """Aplica stres asupra populatiei.
        Functia de stres, se aplica atunci cand ajungem intr-un minim local,
        cauta cele mai frecvente secvente de genom si aplica un stres modifica acele zone
        population   - populatia
        best_individ - individul cu cel mai bun fitness
        best_score   - cea mai buna distanta
        """
        raise NameError("Nu este implementata functia 'permuteSimilarIndivids'!!!")

    def getArgsWeaks(self, fitness_values, size):
        """Returneaza pozitiile 'size' cu cele mai mici valori, ale fitnesului
        fitness_values - valorile fitness a populatiei
        size           - numarul de argumente cu cei mai buni indivizi
        """
        args = np.argpartition(fitness_values, size)
        return args[:size]

    def getArgsElite(self, fitness_values):
        """Returneaza pozitiile 'ELITE_SIZE' cu cele mai mari valori, ale fitnesului
        fitness_values - valorile fitness a populatiei
        """
        args = np.argpartition(fitness_values,-self.ELITE_SIZE)
        args = args[-self.ELITE_SIZE:]
        return args
