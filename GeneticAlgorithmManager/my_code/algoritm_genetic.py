#!/usr/bin/python

import numpy as np
import yaml
import sys
import warnings

def remove_modules(modules_name, *arg):
    if (modules_name in sys.modules):
        del sys.modules[modules_name]
        for key in arg:
            tmp_modules = "{}.{}".format(modules_name, key)
            del sys.modules[tmp_modules]

remove_modules("root_GA")
remove_modules("genoms")
remove_modules("callback")
remove_modules("crossover")
remove_modules("fitness")
remove_modules("init_population")
remove_modules("metrics")
remove_modules("mutate")
remove_modules("select_parent")

from root_GA import *
from genoms import *
from callback import *
from crossover import *
from fitness import *
from init_population import *
from metrics import *
from mutate import *
from select_parent import *

class GeneticAlgorithm(RootGA):
    """
    Managerul de configuratie al algoritmului genetic,
    """
    def __init__(self, name="", extern_commnad_file="", **configs):
        super().__init__()
        self.__name = name
        self.__last_mutation_rate  = None
        self.__extern_commnad_file = extern_commnad_file
        self.__is_stop = False
        self.unpackConfig(**configs)
        self.initByConfig()

    def __str__(self):
        str_info = "Name: {}\n{}".format(self.__name, super().__str__())
        str_info += "\nExtern comand filename '{}'".format(self.__extern_commnad_file)
        str_info += "\nConfigs: {}".format(self.__configs)
        for function in self.__functions:
            str_info += "\n{}".format(str(function))
        str_info += "\n"
        return str_info

    def __call__(self):
        """
        Manager pentru a face sincronizarea dintre toate functionalele
        """
        self.__init_command()
        # initiaizarea populatiei
        if (self.__genoms.is_genoms()==False):
            self.initPopulation(self.POPULATION_SIZE)
        # calculate metrics
        metric_values  = self.metrics(self.__genoms)
        # init fitness value
        fitness_values = self.fitness(metric_values)
        # obtinerea pozitiei pentru elite
        args_elite = self.getArgsElite(fitness_values)
        # evolutia generatiilor
        for generation in range(self.GENERATIONS):
            # pentru oprire fortata
            if (self.__is_stop):
                break
            # start selectie populatie
            self.selectParent1.startEpoch(fitness_values)
            self.selectParent2.startEpoch(fitness_values)
            for _ in range(self.POPULATION_SIZE):
                # selectarea positia parinte 1
                arg_parent1 = self.selectParent1()
                # selectarea positia parinte 1
                arg_parent2 = self.selectParent2()
                # obtinerea parintilor
                parent1 = self.__genoms[arg_parent1]
                parent2 = self.__genoms[arg_parent2]
                # incrucisarea parintilor
                offspring = self.crossover(parent1, parent2)
                # mutatii
                offspring = self.mutate(parent1, parent2, offspring) # in_place operation
                # adauga urmasii la noua generatie
                self.__genoms.append(offspring)
            # obtinerea indivizilor ce fac parte din elita
            genome_elites  = self.__genoms[args_elite]
            if (self.__update_elite_fitness):
                fitness_elites = fitness_values[args_elite]
            else:
                fitness_elites = None
            # schimbarea generatiei
            self.__genoms.save()
            # calculate metrics
            metric_values  = self.metrics(self.__genoms)
            # calculare fitness
            fitness_values = self.fitness(metric_values)
            # adaugare elita in noua populatie
            self.setElitesByFitness(fitness_values, genome_elites, fitness_elites)
            # obtinerea pozitiei pentru elite
            args_elite = self.getArgsElite(fitness_values)
            # calculare metrici
            scores = self.metrics.getScore(self.__genoms, fitness_values)
            # adaugare stres in populatie atunci cand lipseste progresul
            self.stres(scores)
            # afisare metrici
            self.showMetrics(generation, scores)
            # salveaza istoricul
            self.callback(generation, scores)
            # evolution
            self.evolutionMonitor(scores)

        return self.metrics.getBestIndivid()

    def __unpackGenomsConfigure(self, str_functia, chromozomes_name, **configs):
        # 
        ret_conf_chromozomes = {}
        if (configs is not None):
            for chromozome_name in chromozomes_name:
                tmp_config = "{}_{}".format(str_functia, chromozome_name)
                chromozome_configs = configs.get(tmp_config, None)
                ret_conf_chromozomes[chromozome_name] = chromozome_configs
                if (chromozome_configs is None):
                    warnings.warn("Lipseste configuratia, pentru chromozomul '{}', functia '{}'".format(chromozome_name, str_functia))
        else:
            warnings.warn("Lipseste 'configs'")

        return ret_conf_chromozomes

    def __unpackConfigure(self, str_functia, **configs):
        method, method_configs = None, {}
        if (configs is not None):
            method_configs = configs.get(str_functia, None)
            if (method_configs is not None):
                method = method_configs.pop("method", None)
            else:
                method_configs = {}
                warnings.warn("Lipseste metoda, pentru functia de '{}'".format(str_functia))
        else:
            warnings.warn("Lipseste 'configs'")

        return method, method_configs

    def initByConfig(self):
        if (self.__configs is not None):
            #
            subset_size = self.__configs.get("subset_size", 5)
            self.__score_evolution = np.zeros(subset_size, dtype=np.float32)
            #
            self.__update_elite_fitness = self.__configs.get("update_elite_fitness", True)
        else:
            subset_size = 5
            self.__score_evolution = np.zeros(subset_size, dtype=np.float32)
            #
            self.__update_elite_fitness = True

    def unpackConfig(self, **configs):
        # salveaza configuratiile
        self.__functions = []
        # configureaza genoms
        config = configs.get("genoms", {})
        chromozomes_name = list(config.keys())
        self.__genoms = Genoms(size=self.GENOME_LENGTH, **config)
        self.__functions.append(self.__genoms)
        # configurare metrici
        method, method_configs = self.__unpackConfigure("metric", **configs)
        self.metrics = Metrics(method, **method_configs)
        self.__functions.append(self.metrics)
        # configurare initializare populatie
        method, method_configs = self.__unpackConfigure("init_population", **configs)
        self.initPopulation = InitPopulation(method, self.metrics, self.__genoms, **method_configs)
        self.__functions.append(self.initPopulation)
        # configurare fitness
        method, method_configs = self.__unpackConfigure("fitness", **configs)
        self.fitness = Fitness(method, **method_configs)
        self.__functions.append(self.fitness)
        # configurate selectie parinti
        method, method_configs = self.__unpackConfigure("select_parent1", **configs)
        self.selectParent1 = SelectParent(method, **method_configs)
        self.__functions.append(self.selectParent1)
        method, method_configs = self.__unpackConfigure("select_parent2", **configs)
        self.selectParent2 = SelectParent(method, **method_configs)
        self.__functions.append(self.selectParent2)
        # configurare incrucisare
        chromozome_configs = self.__unpackGenomsConfigure("crossover", chromozomes_name, **configs)
        print("chromozome_configs", chromozome_configs)
        self.crossover = Crossover(self.__genoms, **chromozome_configs)
        self.__functions.append(self.crossover)
        # configurare mutatie
        chromozome_configs = self.__unpackGenomsConfigure("mutate", chromozomes_name, **configs)
        self.mutate   = Mutate(self.__genoms, **chromozome_configs)
        self.__functions.append(self.mutate)
        # configurare callback salvare, istoricul de antrenare
        config        = configs.get("callback", {})
        self.callback = Callback(**config)
        self.__functions.append(self.callback)
        # configurare manager salvare, istoricul de antrenare
        self.__configs = configs.get("manager", None)


    def help(self):
        info  = "'nume': numele obiectului\n"
        info += "'extern_commnad_file': numele fisierului in care vor fi adaugate comenzile externe, (oprire fortata = stop=True)\n"
        info += "'manager': {\"subset_size\":5, \"update_elite_fitness\": True}\n"
        info += "'genoms': "+self.__genoms.help()
        info += "'metric': "+self.metrics.help()
        info += "'init_population': "+self.initPopulation.help()
        info += "'fitness': "+self.fitness.help()
        info += "'select_parent': {'select_parent1': 'select_parent2'}: "+self.selectParent1.help()
        info += "'crossover': "+self.crossover.help()
        info += "'mutate': "+self.mutate.help()
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
        GENOME_LENGTH   = kw.get("GENOME_LENGTH", None)
        POPULATION_SIZE = kw.get("POPULATION_SIZE", None)
        if (GENOME_LENGTH is not None):
            self.__genoms.setSize(GENOME_LENGTH)
        if (POPULATION_SIZE is not None):
            self.__genoms.setPopulationSize(POPULATION_SIZE)


    def evolutionMonitor(self, evolution_scores):
        """
        Monitorizarea evolutiei de invatare: datele sunt pastrate intr-un vector
        evolution_scores - scorul evolutiei
        """
        self.__score_evolution[:-1] = self.__score_evolution[1:]
        self.__score_evolution[-1]  = evolution_scores["score"]
        if (evolution_scores["best_fitness"] < 0):
            raise Exception("Best fitness is '0'")

    # Fitness.__call__ always requires BOTH population AND metric_values.
    # => we must compute metrics first, then compute fitness again.
    def setElites(self, elites):
        if (self.__genoms.is_genoms()==False):
            self.initPopulation(self.POPULATION_SIZE)

        if (elites.shape[0] > 0):
            # MUST compute metrics first
            metric_values  = self.metrics(self.__genoms)
            fitness_values = self.fitness(metric_values)
            # set elites
            args = self.getArgsWeaks(fitness_values, elites.shape[0])
            self.__genoms[args] = elites

    def setElitesByFitness(self, fitness_values, elites, fitness_elites=None):
        if (self.__genoms.is_genoms()==False):
            self.initPopulation(self.POPULATION_SIZE)

        # here fitness_values already exists (caller passed it)
        if (elites.shape[0] > 0):
            args = self.getArgsWeaks(fitness_values, elites.shape[0])
            self.__genoms[args]  = elites
            if (fitness_elites is not None):
                fitness_values[args] = fitness_elites

    def showMetrics(self, generation, d_info):
        """Afisare metrici"""
        metric_info ="Name:{}, Generatia: {}, ".format(self.__name, generation)
        for key in d_info.keys():
            val = d_info[key]
            if (isinstance(val, float)):
                val = round(val, 3)
            metric_info +="{}: {}, ".format(key, val)
        print(metric_info)

    def stres(self, evolution_scores):
        """Aplica stres asupra populatiei.
        Functia de stres, se aplica atunci cand ajungem intr-un minim local,
        cauta cele mai frecvente secvente de genom si aplica un stres modifica acele zone
        evolution_scores - scorul evolutiei
        """
        check_distance = np.allclose(self.__score_evolution.mean(), evolution_scores["score"], rtol=1e-01, atol=1e-03)
        #print("distance evolution {}, distance {}".format(check_distance, best_distance))
        if (check_distance):
            self.__score_evolution[:] = 0
            self.__last_mutation_rate = self.MUTATION_RATE
            print("evolution_scores {}".format(evolution_scores))
            self.setParameters(MUTATION_RATE=1.)
            #self.mutate.increaseSubsetSize()
            self.externCommand()
        else:
            #self.mutate.decreaseSubsetSize()
            if (self.__last_mutation_rate is not None):
                self.setParameters(MUTATION_RATE=self.__last_mutation_rate)

    def getArgsWeaks(self, fitness_values, size):
        """Returneaza pozitiile 'size' cu cele mai mici valori, ale fitnesului
        fitness_values - valorile fitness a populatiei
        size           - numarul de argumente cu cei mai buni indivizi
        """
        if (size > 0):
            args = np.argpartition(fitness_values, size)[:size]
        else:
            args = np.array([], dtype=np.int32)
        return args

    def getArgsElite(self, fitness_values):
        """Returneaza pozitiile 'ELITE_SIZE' cu cele mai mari valori, ale fitnesului
        fitness_values - valorile fitness a populatiei
        """
        if (self.ELITE_SIZE > 0):
            args = np.argpartition(fitness_values,-self.ELITE_SIZE)
            args = args[-self.ELITE_SIZE:]
        else:
            args = np.array([], dtype=np.int32)
        return args

    def externCommand(self):
        command_dict = self.__read_command_yaml_file()
        self.__is_stop = command_dict["stop"]

    def __read_command_yaml_file(self):
        if (isinstance(self.__extern_commnad_file, str) and (Path(self.__extern_commnad_file).is_file())):
            with open(self.__extern_commnad_file) as file :
                # The FullLoader parameter handles the conversion from YAML
                # scalar values to Python the dictionary format
                command_dict = yaml.load(file, Loader=yaml.FullLoader)
        else :
            command_dict = {"stop":False}
        return command_dict

    def __init_command(self) :
        if (isinstance(self.__extern_commnad_file, str) and (not Path(self.__extern_commnad_file).is_file())):
            # create a config file
            Path(self.__extern_commnad_file).touch(mode=0o666, exist_ok=True)
        if (isinstance(self.__extern_commnad_file, str) and (Path(self.__extern_commnad_file).is_file())):
            # save default rating in yaml file
            commands      = "stop : {}".format(False)
            yaml_commands = yaml.safe_load(commands)

            with open(self.__extern_commnad_file, "w") as file :
                yaml.dump(yaml_commands, file)
