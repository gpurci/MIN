#!/usr/bin/python
import numpy as np
import yaml
import sys
import warnings
from pathlib import Path


# ============================================================
#   REMOVE MODULES (HOT RELOAD)
# ============================================================
def remove_modules(modules_name, *arg):
    if modules_name in sys.modules:
        del sys.modules[modules_name]
    for key in arg:
        tmp_modules = f"{modules_name}.{key}"
        if tmp_modules in sys.modules:
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


# ============================================================
#   IMPORTS AFTER CLEANING
# ============================================================
from root_GA import *
from genoms import *
from callback import *
from crossover import *
from fitness import *
from init_population import *
from metrics import *
from mutate import *
from select_parent import *


# ============================================================
#              GENETIC ALGORITHM MANAGER
# ============================================================
class GeneticAlgorithm(RootGA):
    """Manager al configuratiei si executiei unui algoritm genetic."""

    # --------------------------------------------------------
    def __init__(self, name="", extern_commnad_file="", **configs):
        super().__init__()
        self.__name = name
        self.__score_evolution = np.zeros(5, dtype=np.float32)
        self.__last_mutation_rate = None
        self.__extern_commnad_file = extern_commnad_file
        self.__is_stop = False

        # ✨ NEW: memetic / LS parameters
        self.LS_START_GEN = 1     # incepe LS dupa generatia 500
        self.LS_FREQ      = 10      # aplica LS o data la 10 generatii
        self.LS_ELITE_N   = 20      # LS pe cei mai buni 20 indivizi
        self.LS_MAX_ITER  = 25      # iteratii 2-opt

        self.setConfig(**configs)

    # --------------------------------------------------------
    def __str__(self):
        str_info = f"Name: {self.__name}\n{super().__str__()}"
        str_info += "\nConfigs:"
        for function in self.__functions:
            str_info += f"\n{function}"
        str_info += "\n"
        return str_info

    # --------------------------------------------------------
    def __call__(self):
        """Manager pentru sincronizarea functionalităților."""

        self.__init_command()

        # Inițializează populația dacă lipsește
        if not self.__genoms.is_genoms():
            self.initPopulation(self.POPULATION_SIZE)

        # Metrici + fitness inițiale
        metric_values = self.metrics(self.__genoms)
        fitness_values = self.fitness(metric_values)

        # Pozițiile elitelor
        args_elite = self.getArgsElite(fitness_values)

        # --------------------------------------------------
        #                EVOLUȚIA GENERAȚIILOR
        # --------------------------------------------------
        for generation in range(self.GENERATIONS):

            if self.__is_stop:
                break

            self.selectParent1.startEpoch(fitness_values)
            self.selectParent2.startEpoch(fitness_values)

            # -------------------------
            #    GENERARE POPULAȚIE
            # -------------------------
            for _ in range(self.POPULATION_SIZE):

                arg_p1 = self.selectParent1()
                arg_p2 = self.selectParent2()

                parent1 = self.__genoms[arg_p1]
                parent2 = self.__genoms[arg_p2]

                # Crossover
                offspring = self.crossover(parent1, parent2)

                # Inject TTP info (pentru mutația TSP)
                try:
                    self.mutate._Mutate__chromozoms["tsp"]["kp_bits"] = offspring["kp"]
                    self.mutate._Mutate__chromozoms["tsp"]["dataset"] = self.dataset
                except:
                    pass

                # Mutatie
                offspring = self.mutate(parent1, parent2, offspring)

                # Add to new generation
                self.__genoms.append(offspring)

            # Elită veche (din generația anterioară)
            genome_elites = {
                key: self.__genoms.chromozomes(key)[args_elite]
                for key in self.__genoms.keys()
            }
            genome_elites = np.core.records.fromarrays(
                [genome_elites[key] for key in self.__genoms.keys()],
                names=",".join(self.__genoms.keys())
            )

            # Finalizare generație
            self.__genoms.save()

            # Metrici + fitness (inainte de LS)
            metric_values = self.metrics(self.__genoms)
            fitness_values = self.fitness(metric_values)

            # Memetic 2-opt pe elite (nu pe toata populatia)
            if (
                generation >= self.LS_START_GEN and
                (generation % self.LS_FREQ == 0)
            ):
                self.local_search_elites(fitness_values)
                # Recalculeaza metrici si fitness dupa LS
                metric_values = self.metrics(self.__genoms)
                fitness_values = self.fitness(metric_values)

            # Reinserare elită anterioară (din generatia t-1)
            self.setElitesByFitness(fitness_values, genome_elites)

            # Pozițiile noilor elite (dupa LS + reinserare)
            args_elite = self.getArgsElite(fitness_values)

            # Metrici generale
            scores = self.metrics.getScore(self.__genoms, fitness_values)

            # stres evolutiv
            self.stres(scores)

            # Afișare
            self.showMetrics(generation, scores)

            # Log
            self.callback(generation, scores)

            # Monitorizare
            self.evolutionMonitor(scores)

        return self.metrics.getBestIndivid()

    # ============================================================
    #         LOCAL SEARCH ON ELITES (MEMETIC STEP)
    # ============================================================
    def local_search_elites(self, fitness_values):
        """
        Aplica 2-opt doar pe cei mai buni LS_ELITE_N indivizi.
        Ruleaza rar (definit de LS_FREQ) si cu max_iter mic.
        """
        n_elite = min(self.LS_ELITE_N, self.POPULATION_SIZE)
        if n_elite <= 0:
            return

        # Indicii celor mai buni indivizi
        elite_args = np.argpartition(fitness_values, -n_elite)[-n_elite:]

        tsp_mat = self.__genoms.chromozomes("tsp")
        dist = self.dataset["distance"]

        improved_count = 0
        for idx in elite_args:
            try:
                route = tsp_mat[idx]
                new_route = self.initPopulation.local_search_2opt(
                    route,
                    dist,
                    max_iter=self.LS_MAX_ITER
                )
                tsp_mat[idx] = new_route
                improved_count += 1
            except Exception as e:
                print(f"[LS elite] error on idx {idx}:", e)

        print(f"[LS elite] applied 2-opt on {improved_count}/{n_elite} elites.")

    # ============================================================
    #               CONFIGURARE AUTOMATĂ
    # ============================================================
    def __unpackGenomsConfigure(self, name, chromozomes_name, **configs):
        ret = {}
        if configs is not None:
            for chrom in chromozomes_name:
                key = f"{name}_{chrom}"
                ret[chrom] = configs.get(key, None)
                if ret[chrom] is None:
                    warnings.warn(
                        f"Lipseste configuratia pentru '{chrom}', functia '{name}'"
                    )
        else:
            warnings.warn("Lipseste 'configs'")
        return ret

    def __unpackConfigure(self, name, **configs):
        if configs is None:
            warnings.warn("Lipseste 'configs'")
            return None, {}

        method_configs = configs.get(name, None)
        if method_configs is None:
            warnings.warn(f"Lipseste metoda pentru functia '{name}'")
            return None, {}

        method = method_configs.pop("method", None)
        return method, method_configs

    # --------------------------------------------------------
    def setConfig(self, **configs):

        self.__functions = []

        # -------------------------------
        # GENOMS
        # -------------------------------
        config = configs.get("genoms", {})
        chromozomes_name = list(config.keys())
        self.__genoms = Genoms(size=self.GENOME_LENGTH, **config)
        self.__functions.append(self.__genoms)

        # -------------------------------
        # METRIC
        # -------------------------------
        method, cfg = self.__unpackConfigure("metric", **configs)
        self.metrics = Metrics(method, **cfg)
        self.__functions.append(self.metrics)

        # -------------------------------
        # INIT POPULATION
        # -------------------------------
        method, cfg = self.__unpackConfigure("init_population", **configs)
        self.initPopulation = InitPopulation(
            method, self.metrics, self.__genoms, **cfg
        )
        self.__functions.append(self.initPopulation)

        # -------------------------------
        # FITNESS
        # -------------------------------
        method, cfg = self.__unpackConfigure("fitness", **configs)
        self.fitness = Fitness(method, **cfg)
        self.__functions.append(self.fitness)

        # -------------------------------
        # SELECT PARENTS
        # -------------------------------
        method, cfg = self.__unpackConfigure("select_parent1", **configs)
        self.selectParent1 = SelectParent(method, **cfg)
        self.__functions.append(self.selectParent1)

        method, cfg = self.__unpackConfigure("select_parent2", **configs)
        self.selectParent2 = SelectParent(method, **cfg)
        self.__functions.append(self.selectParent2)

        # -------------------------------
        # CROSSOVER
        # -------------------------------
        chrom_cfg = self.__unpackGenomsConfigure(
            "crossover", chromozomes_name, **configs
        )
        self.crossover = Crossover(self.__genoms, **chrom_cfg)
        self.__functions.append(self.crossover)

        # -------------------------------
        # MUTATION
        # -------------------------------
        chrom_cfg = self.__unpackGenomsConfigure(
            "mutate", chromozomes_name, **configs
        )
        self.mutate = Mutate(self.__genoms, **chrom_cfg)
        self.__functions.append(self.mutate)

        # -------------------------------
        # CALLBACK
        # -------------------------------
        config = configs.get("callback", {})
        self.callback = Callback(**config)
        self.__functions.append(self.callback)

    # ============================================================
    #                  SET DATASET / PARAMETERS
    # ============================================================
    def setDataset(self, dataset):
        self.dataset = dataset
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

        GENOME_LENGTH = kw.get("GENOME_LENGTH", None)
        POPULATION_SIZE = kw.get("POPULATION_SIZE", None)

        if GENOME_LENGTH is not None:
            self.__genoms.setSize(GENOME_LENGTH)

        if POPULATION_SIZE is not None:
            self.__genoms.setPopulationSize(POPULATION_SIZE)

    # ============================================================
    #                    EVOLUTION MONITOR
    # ============================================================
    def evolutionMonitor(self, evolution_scores):
        self.__score_evolution[:-1] = self.__score_evolution[1:]
        self.__score_evolution[-1] = evolution_scores["score"]

        if evolution_scores["best_fitness"] <= 0:
            raise Exception("Best fitness is '0'")

    # ============================================================
    #                          ELITES
    # ============================================================
    def setElites(self, elites):
        if not self.__genoms.is_genoms():
            self.initPopulation(self.POPULATION_SIZE)

        metric_values = self.metrics(self.__genoms)
        fitness_values = self.fitness(metric_values)

        args = self.getArgsWeaks(fitness_values, self.ELITE_SIZE)
        self.__genoms[args] = elites

    def setElitesByFitness(self, fitness_values, elites):
        if not self.__genoms.is_genoms():
            self.initPopulation(self.POPULATION_SIZE)

        args = self.getArgsWeaks(fitness_values, elites.shape[0])
        self.__genoms[args] = elites

    # ============================================================
    #                          DISPLAY
    # ============================================================
    def showMetrics(self, generation, d_info):
        metric_info = f"Name:{self.__name}, Generatia:{generation}, "
        for key, val in d_info.items():
            if isinstance(val, float):
                val = round(val, 3)
            metric_info += f"{key}: {val}, "
        print(metric_info)

    # ============================================================
    #                           STRES
    # ============================================================
    def stres(self, evolution_scores):

        score_now = evolution_scores["score"]
        score_mean = self.__score_evolution.mean()

        # Update sliding window
        self.__score_evolution[:-1] = self.__score_evolution[1:]
        self.__score_evolution[-1] = score_now

        plateau = np.allclose(score_now, score_mean, rtol=1e-3, atol=1e-8)

        # Get reference to fitness config
        fit_cfg = self.fitness._Fitness__configs

        if plateau:
            # ----------------------------
            # Plateau detected
            # ----------------------------
            print("⚠️ Plateau detected → boosting mutation + selective pressure")

            # Boost mutation temporarily
            if self.__last_mutation_rate is None:
                self.__last_mutation_rate = self.MUTATION_RATE
            self.setParameters(MUTATION_RATE=0.8)

            # Increase alpha (selective pressure)
            old_alpha = fit_cfg.get("alpha", 2.0)
            new_alpha = min(6.0, old_alpha + 0.5)
            fit_cfg["alpha"] = new_alpha

            print(f"   • alpha increased: {old_alpha} → {new_alpha}")

            # Reset stagnation history
            self.__score_evolution[:] = score_now

        else:
            # ----------------------------
            # No plateau → restore values
            # ----------------------------
            if self.__last_mutation_rate is not None:
                self.setParameters(MUTATION_RATE=self.__last_mutation_rate)

            # Decay alpha slowly back to baseline
            base_alpha = 2.0
            cur_alpha = fit_cfg.get("alpha", base_alpha)
            if cur_alpha > base_alpha:
                fit_cfg["alpha"] = max(base_alpha, cur_alpha - 0.2)

    # ============================================================
    #                          RANKING
    # ============================================================
    def getArgsWeaks(self, fitness_values, size):
        args = np.argpartition(fitness_values, size)
        return args[:size]

    def getArgsElite(self, fitness_values):
        args = np.argpartition(fitness_values, -self.ELITE_SIZE)
        return args[-self.ELITE_SIZE:]

    # ============================================================
    #                   EXTERN COMMAND / YAML
    # ============================================================
    def externCommand(self):
        command_dict = self.__read_command_yaml_file()
        self.__is_stop = command_dict["stop"]

    def __read_command_yaml_file(self):

        if (
            isinstance(self.__extern_commnad_file, str)
            and Path(self.__extern_commnad_file).is_file()
        ):
            with open(self.__extern_commnad_file) as file:
                return yaml.load(file, Loader=yaml.FullLoader)

        return {"stop": False}

    def __init_command(self):
        if (
            isinstance(self.__extern_commnad_file, str)
            and not Path(self.__extern_commnad_file).is_file()
        ):
            Path(self.__extern_commnad_file).touch(mode=0o666, exist_ok=True)

        if Path(self.__extern_commnad_file).is_file():
            yaml_commands = yaml.safe_load("stop: False")
            with open(self.__extern_commnad_file, "w") as file:
                yaml.dump(yaml_commands, file)
