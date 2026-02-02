#!/usr/bin/python
"""
LocalSearchFactory / registry for all local-search operators.

Usage examples:

# 1) Single operator by name
ls = LocalSearchFactory("hybrid_2opt", dataset, iters=120)

# 2) Chain operators by string
ls = LocalSearchFactory("hybrid_2opt | or_opt_restrict | three_opt_restrict", dataset)

# 3) Chain by list + per-op configs
ls = LocalSearchFactory(
    ["two_opt_distance", "or_opt_restrict"],
    dataset,
    per_op_configs={
        "two_opt_distance": {"subset_size": 10},
        "or_opt_restrict": {"iters": 1}
    }
)

Returned object is callable GA-style:
    improved = ls(parent1, parent2, offspring)

So it plugs into your elite-search wrapper directly.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Union

from extension.local_search_algorithms.two_opt import TwoOpt
from extension.local_search_algorithms.or_opt import OrOpt
from extension.local_search_algorithms.three_opt import ThreeOpt
from extension.local_search_algorithms.kp_greedy import KPGreedyImprove
from extension.local_search_algorithms.vnd import VND
from extension.local_search_algorithms.tabu_hybrid_search import TabuHybridSearch
from extension.local_search_algorithms.ttp_vnd import TTPVNDLocalSearch


# ---------------------------------------------------------------------
# Chain wrapper (applies multiple local searches sequentially)
# ---------------------------------------------------------------------
@dataclass
class ChainLocalSearch:
    ops: List[object]

    def setParameters(self, **kw):
        for op in self.ops:
            if hasattr(op, "setParameters"):
                op.setParameters(**kw)

    def __call__(self, p1, p2, offspring, **kw):
        x = offspring
        for op in self.ops:
            x = op(p1, p2, x, **kw) if callable(op) else x
        return x


# ---------------------------------------------------------------------
# Registry
# Map name -> (class, default_method)
# ---------------------------------------------------------------------
_REGISTRY = {
    "two_opt":              (TwoOpt, "two_opt"),
    "two_opt_rand":         (TwoOpt, "two_opt_rand"),
    "two_opt_distance":     (TwoOpt, "two_opt_distance"),
    "two_opt_LS":           (TwoOpt, "two_opt_LS"),
    "or_opt":               (OrOpt, "or_opt"),
    "three_opt_restrict":   (ThreeOpt, "three_opt_restrict"),
    "hybrid_2opt":          (TabuHybridSearch, "hybrid_2opt"),

    # KP / TTP-aware
    "kp_greedy":            (KPGreedyImprove, None),

    # Composite
    "vnd_LS":               (VND, None),
    "vnd_ttp":              (TTPVNDLocalSearch, None),
}

# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
def LocalSearchFactory(
    name: Union[str, List[str]],
    dataset: dict,
    per_op_configs: Dict[str, Dict[str, Any]] = None,
    **configs
):
    """
    Create a local search operator by name or chain.

    Parameters
    ----------
    name:
        - str: one operator name OR a pipe-separated chain:
              "hybrid_2opt | or_opt_restrict"
        - list[str]: chain of operator names
    dataset:
        TTP/TSP dataset dict containing at least "distance",
        and for kp_greedy also item_profit/item_weight.
    per_op_configs:
        Optional dict of per-operator configs in a chain.
    configs:
        Default configs passed to ALL ops unless overridden per_op_configs.
    """
    per_op_configs = per_op_configs or {}

    # normalize to list of names
    if isinstance(name, str):
        parts = [p.strip() for p in name.split("|") if p.strip()]
    else:
        parts = list(name)

    if len(parts) == 0:
        raise ValueError("LocalSearchFactory got empty name/chain.")

    # if single op, return it directly (no chain wrapper)
    if len(parts) == 1:
        key = parts[0]
        if key not in _REGISTRY:
            raise KeyError(f"Unknown local search '{key}'. Known: {list(_REGISTRY.keys())}")

        cls, method = _REGISTRY[key]
        cfg = dict(configs)
        cfg.update(per_op_configs.get(key, {}))

        # ----- SPECIAL CASE: TTP-level VND -----
        if cls is TTPVNDLocalSearch:
            return cls(dataset=dataset, **cfg)

        # ----- Standard Operators (route-only) -----
        if method is None:
            # e.g. KPGreedyImprove, VND
            if "dataset" in cls.__init__.__code__.co_varnames:
                return cls(dataset=dataset, **cfg)
            return cls(dataset, **cfg)

        # ----- Operators that use GA-style method parameter -----
        return cls(method, dataset, **cfg)

    # otherwise build chain
    ops = []
    for key in parts:
        if key not in _REGISTRY:
            raise KeyError(f"Unknown local search '{key}'. Known: {list(_REGISTRY.keys())}")

        cls, method = _REGISTRY[key]
        cfg = dict(configs)
        cfg.update(per_op_configs.get(key, {}))

        if method is None:
            op = cls(dataset=dataset, **cfg) if "dataset" in cls.__init__.__code__.co_varnames else cls(dataset, **cfg)
        else:
            op = cls(method, dataset, **cfg)

        ops.append(op)

    return ChainLocalSearch(ops)
