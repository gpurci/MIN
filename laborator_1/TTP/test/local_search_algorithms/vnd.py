#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import RootGA

from extension.local_search_algorithms.two_opt import TwoOpt
from extension.local_search_algorithms.or_opt import OrOpt
from extension.local_search_algorithms.three_opt import ThreeOpt


class VND(RootGA):
    """
    Variable Neighborhood Descent over TSP routes.

    Compatible with:
        - InitPopulationHybrid: vnd(None, None, route)
        - LocalSearchFactory("vnd_LS", dataset, max_rounds=...)
        - GA-style signature: op(p1, p2, route, **cfg)
    """

    def __init__(self, dataset, ops=None, **configs):
        super().__init__()

        if dataset is None:
            raise ValueError("VND requires dataset")

        self.dataset  = dataset
        self.distance = dataset["distance"]
        self._configs = configs  # e.g. max_rounds, etc.

        if ops is None:
            ops = [
                TwoOpt("two_opt_LS", dataset),
                OrOpt("or_opt_restrict", dataset),
                ThreeOpt("three_opt_restrict", dataset),
            ]
        self.ops = ops

    def __str__(self):
        return f"VND(ops={[op.__class__.__name__ for op in self.ops]}, configs={self._configs})"

    def setParameters(self, **kw):
        super().setParameters(**kw)
        for op in self.ops:
            if hasattr(op, "setParameters"):
                op.setParameters(**kw)

    @staticmethod
    def _route_len(route, dist):
        return dist[route[:-1], route[1:]].sum() + dist[route[-1], route[0]]

    def __call__(self, p1, p2, route, **call_cfg):
        cfg = dict(self._configs)
        cfg.update(call_cfg)

        max_rounds = cfg.get("max_rounds", 5)

        dist    = self.distance
        cur     = route.copy()
        cur_len = self._route_len(cur, dist)

        for _ in range(max_rounds):
            improved = False

            for op in self.ops:
                if hasattr(op, "twoOptLS"):
                    new_route = op.twoOptLS(cur)
                else:
                    new_route = op(p1, p2, cur)

                new_len = self._route_len(new_route, dist)

                if new_len + 1e-12 < cur_len:
                    cur     = new_route
                    cur_len = new_len
                    improved = True

            if not improved:
                break

        return cur
