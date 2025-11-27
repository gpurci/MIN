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
    "generations": 1000,
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
GA_SIMPLE = _cfg(
    BASE_GA,
    id="ga_simpl",
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

