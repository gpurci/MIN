# builder_ttp.py

from extension.init_population.my_code.init_hybrid_population_MATEI import InitPopulationHybrid
from extension.crossover.my_code.crossover_tsp_MATEI import CrossoverMateiTSP
from extension.crossover.my_code.crossover_kp_MATEI import CrossoverMateiKP
from extension.mutate.my_code.mutate_tsp_MATEI import MutateMateiTSP
from extension.mutate.my_code.mutate_kp_MATEI import MutateMateiKP

from extension.kp_repair.cheap_mutate_repair import (
    MutateKPWithRepairAndStress,  # stress-aware
    MutateKPWithRepair,           # alias = no-stress version
)
from extension.kp_repair.multi_kp_repair import MultiStrategyKPRepair
from extension.kp_repair.cheap_kp_repair import FastKPRepair

from extension.fitness.my_code.fitness_ttpMATEI import TTPFitness
from extension.metrics.my_code.metrics_ttpMATEI import MetricsTTPMATEI
from extension.stres.my_code.stres_ttp_MATEI import StresTTP
from extension.local_search_algorithms.ttp_vnd import TTPVNDLocalSearch

from extension.ga_elite_local_search import GeneticAlgorithmWithEliteSearch
from extension.select_parent.my_code.select_parent import SelectParent

def build_ttp_ga(dataset, cfg, extern_command_file, log_file):
    """
    Build a TTP GA instance according to 'cfg'.

    Required keys in 'dataset':
        - "GENOME_LENGTH", "item_weight", "item_profit",
          "distance", "W", "R", "v_min", "v_max"

    Required keys in 'cfg':
        - id, population_size, generations,
          mutation_rate, crossover_rate
        - use_multi_repair, use_stress, use_vnd, use_tsp_ls, use_kp_ls
        - (optional) elite_size, elite_freq
    """

    # ---- init & operators ----
    init_population_ttp_obj = InitPopulationHybrid(method="TTP_hybrid", dataset=dataset)

    crossover_tsp_obj = CrossoverMateiTSP("mixt", p_select=[0.7, 0.3])

    base_crossover_kp = CrossoverMateiKP("mixt", p_select=[0.1, 0.1, 0.1, 0.3, 0.4])

    mutate_tsp_obj = MutateMateiTSP(
        "mixt",
        dataset=dataset,
        subset_size=40,
        p_select=[0.4, 0.3, 0.3],
    )

       # ---- repair (multi or cheap) ----
    if cfg["use_multi_repair"]:
        # allow each config to override the p_modes of MultiStrategyKPRepair
        multi_p_modes = cfg.get("multi_p_modes", (0.4, 0.4, 0.2))

        repair = MultiStrategyKPRepair(
            w        = dataset["item_weight"],
            v        = dataset["item_profit"],
            Wmax     = dataset["W"],
            distance = dataset["distance"],
            beta     = 1.0,
            p_modes  = multi_p_modes,
            # you can also tweak this if you want multi-mode only for big overloads
            min_rel_over_for_multi=0.05,
        )
    else:
        # Simple fast repair (no multi-strategy)
        repair = FastKPRepair(
            w    = dataset["item_weight"],
            v    = dataset["item_profit"],
            Wmax = dataset["W"],
        )

    # ---- KP crossover + repair wrapper ----
    from extension.kp_repair.cheap_crossover_repair import CrossoverKPWithRepair
    crossover_kp_obj = CrossoverKPWithRepair(
        base_crossover = base_crossover_kp,
        repair         = repair,
        W              = dataset["W"],
        dataset        = dataset,
    )

    # ---- KP mutation (base) ----
    mutate_kp_base = MutateMateiKP(
        method="mixt_extended",
        rate=0.035,
        p_select=[0.4, 0.3, 0.3],
    )

    # ---- KP mutation (wrapped with repair / stress) ----
    if cfg["use_stress"]:
        # Stress-aware version (StresTTP will toggle stress_mode)
        mutate_kp_obj = MutateKPWithRepairAndStress(
            base_mutator=mutate_kp_base,
            W=dataset["W"],
            profits=dataset["item_profit"],
            weights=dataset["item_weight"],
            stress_destroy_frac=0.35,
            stress_destroy_worst_frac=0.15,
            allow_add=True,
            repair=repair,
        )
    else:
        # Plain mutate + repair, no stress: alias uses stress_mode=False
        mutate_kp_obj = MutateKPWithRepair(
            base_mutator=mutate_kp_base,
            W=dataset["W"],
            profits=dataset["item_profit"],
            weights=dataset["item_weight"],
            allow_add=True,
            repair=repair,
        )

    fitness_obj = TTPFitness("TTP_standard", R=dataset["R"], W=dataset["W"])
    metrics_obj = MetricsTTPMATEI(
        "TTP_linear",
        dataset,
        v_min=dataset["v_min"],
        v_max=dataset["v_max"],
        W=dataset["W"],
    )

    # ---- stress ----
    if cfg["use_stress"]:
        stres_obj = StresTTP(
            plateau_window=15,
            plateau_rtol=1e-3,
            replace_ratio=0.20,
            mutation_boost=True,
            mutation_boost_factor=1.25,
            mutation_rate_max=0.12,
            dynamic_alpha=True,
            restart_ratio=0.40,   # <--- NEW: fraction of pop to restart on strong plateau
        )
        stres_obj.fitness         = fitness_obj
        stres_obj.mutate_kp       = mutate_kp_obj
        stres_obj.init_population = init_population_ttp_obj  # <--- NEW: hybrid init hook
    else:
        from extension.stres.my_code.stres_base import StresBase
        stres_obj = StresBase()


    # ---- VND local search ----
    if cfg["use_vnd"]:
        elite_search_obj = TTPVNDLocalSearch(
            dataset=dataset,
            v_max=dataset["v_max"],
            v_min=dataset["v_min"],
            W=dataset["W"],
            R=dataset["R"],
            max_rounds=5,
            use_kp_ls=cfg["use_kp_ls"],
            use_tsp_ls=cfg["use_tsp_ls"],
        )
    else:
        elite_search_obj = None

    sel_parent1_obj = SelectParent("tour", size_subset=3)
    sel_parent2_obj = SelectParent("tour_choice", size_subset=3)

    # ---- GA wrapper ----
    ttp = GeneticAlgorithmWithEliteSearch(
        name=cfg["id"],
        extern_commnad_file=extern_command_file,
        manager={
            "subset_size": 5,
            "update_elite_fitness": True,
            "select_rate_max": 0.85,
            "select_rate_min": 0.35,
            "select_rate_schedule": "linear",
        },
        genoms={
            "check_freq": 50,
            "tsp": (0, dataset["GENOME_LENGTH"]),
            "kp":  (0, dataset["item_profit"].shape[0]),
        },
        init_population=init_population_ttp_obj,
        metric=metrics_obj,
        elite_search=elite_search_obj,
        elite_chrom="",
        fitness=fitness_obj,
        select_parent1=sel_parent1_obj,
        select_parent2=sel_parent2_obj,
        crossover={
            "method": "chromosome",
            "tsp": crossover_tsp_obj,
            "kp":  crossover_kp_obj,
        },
        mutate={
            "method": "chromosome",
            "tsp": mutate_tsp_obj,
            "kp":  mutate_kp_obj,
        },
        stres=stres_obj,
        callback={"filename": log_file, "freq": 1},
    )

    # ---- global GA parameters ----
    ttp.setParameters(
        GENOME_LENGTH   = dataset["GENOME_LENGTH"],
        GENERATIONS     = cfg["generations"],
        POPULATION_SIZE = cfg["population_size"],
        MUTATION_RATE   = cfg["mutation_rate"],
        CROSSOVER_RATE  = cfg["crossover_rate"],
        SELECT_RATE     = 0.5,           # base; manager overrides adaptively
        dataset         = dataset,
        W               = dataset["W"],
        v_min           = dataset["v_min"],
        v_max           = dataset["v_max"],
        R               = dataset["R"],
        ELITE_SIZE      = cfg.get("elite_size", 40),
        ELITE_FREQ      = cfg.get("elite_freq", 8),
    )

    return ttp


# ------------------------------------------------------------------
# Base dataset & GA defaults
# ------------------------------------------------------------------

BASE_DATASET = {
    "name": "a280-TTP",
    "W": 25936,
    "v_min": 0.1,
    "v_max": 1.0,
    "R": 5.61,
    "GENOME_LENGTH": 280,
}

BASE_GA = {
    "population_size": 400,
    "generations": 150,
    "mutation_rate": 0.08,
    "crossover_rate": 0.70,

    "elite_size": 40,
    "elite_freq": 8,

    "use_multi_repair": True,
    "use_stress": True,
    "use_vnd": True,
    "use_tsp_ls": True,
    "use_kp_ls": True,
}


def _cfg(base, **kwargs):
    c = base.copy()
    c.update(kwargs)
    return c


# ------------------------------------------------------------------
# Config variants (for ablations / experiments)
# ------------------------------------------------------------------

GA_SIMPLE = _cfg(
    BASE_GA,
    id="ga_simple",
    description="Simple GA: cheap repair only, no stress, no VND LS",

    use_multi_repair=False,
    use_stress=False,
    use_vnd=False,
    use_tsp_ls=False,
    use_kp_ls=False,
)

GA_SIMPLE_VND = _cfg(
    BASE_GA,
    id="ga_simple_vnd",
    description="GA + VND (TSP+KP) with cheap repair, no stress",

    use_multi_repair=False,
    use_stress=False,
    use_vnd=True,
    use_tsp_ls=True,
    use_kp_ls=True,
)

GA_FULL = _cfg(
    BASE_GA,
    id="ga_full",
    description="Full method: GA + multi-repair + stress + VND (TSP+KP LS)",

    use_multi_repair=True,
    use_stress=True,
    use_vnd=True,
    use_tsp_ls=True,
    use_kp_ls=True,
)

GA_NO_MULTI_REPAIR = _cfg(
    BASE_GA,
    id="ga_no_multi_repair",
    description="Ablation: multi-repair OFF, stress + VND ON",

    use_multi_repair=False,
    use_stress=True,
    use_vnd=True,
    use_tsp_ls=True,
    use_kp_ls=True,
)

GA_NO_STRESS = _cfg(
    BASE_GA,
    id="ga_no_stress",
    description="Ablation: stress OFF, multi-repair + VND ON",

    use_multi_repair=True,
    use_stress=False,
    use_vnd=True,
    use_tsp_ls=True,
    use_kp_ls=True,
)

GA_NO_TSP_LS = _cfg(
    BASE_GA,
    id="ga_no_tsp_ls",
    description="Ablation: TSP LS OFF, KP LS ON (multi-repair + stress ON)",

    use_multi_repair=True,
    use_stress=True,
    use_vnd=True,
    use_tsp_ls=False,
    use_kp_ls=True,
)

GA_NO_KP_LS = _cfg(
    BASE_GA,
    id="ga_no_kp_ls",
    description="Ablation: KP LS OFF, TSP LS ON (multi-repair + stress ON)",

    use_multi_repair=True,
    use_stress=True,
    use_vnd=True,
    use_tsp_ls=True,
    use_kp_ls=False,
)

GA_BEST_CANDIDATE = _cfg(
    BASE_GA,
    id="ga_best_candidate",
    description="No multi-repair, stress ON, VND with KP-LS only",

    use_multi_repair=False,
    use_stress=True,
    use_vnd=True,
    use_tsp_ls=False,   # <--- off
    use_kp_ls=True,     # <--- on
)

GA_MULTI_GENTLE = _cfg(
    BASE_GA,
    id="ga_multi_gentle",
    description=(
        "GA with gentle multi-repair: mostly FAST, a bit of HORIZON, "
        "stress-coordinated; TSP+KP LS ON."
    ),

    use_multi_repair=True,
    use_stress=True,
    use_vnd=True,
    use_tsp_ls=True,
    use_kp_ls=True,

    # gentle multi-repair: mostly FAST, some HORIZON, no AGGRESSIVE
    multi_p_modes=(0.8, 0.2, 0.0),
)
