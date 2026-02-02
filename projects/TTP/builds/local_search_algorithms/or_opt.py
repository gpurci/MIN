#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import RootGA


class OrOpt(RootGA):
    """
    Or-Opt local search for TSP routes (block relocation).
    Moves a block of length 1..block_max to another position.

    Methods:
        "or_opt"          : deterministic best-improvement
        "or_opt_rand"     : randomized first-improvement
        "or_opt_restrict" : restricted reinsertion using k-nearest neighbors

    Configs:
        block_max=3
        iters=1
        trials=80           (rand)
        k_candidates=15     (restrict)
        seed=None
    """

    def __init__(self, method="or_opt", dataset=None, **configs):
        super().__init__()
        if dataset is None:
            raise ValueError("OrOpt requires dataset with 'distance'.")
        self.dataset = dataset
        self.distance = dataset["distance"]
        self.__method = method
        self.__configs = configs
        self.__fn = self.__unpack(method)
        self._candidates = None  # lazy nearest-neighbor lists

    def __str__(self):
        return f"OrOpt(method={self.__method}, configs={self.__configs})"

    def help(self):
        print("""OrOpt:
    method: 'or_opt' | 'or_opt_rand' | 'or_opt_restrict'
    configs:
        block_max=3, iters=1
        trials=80 (rand), k_candidates=15 (restrict), seed=None
""")

    def __unpack(self, method):
        table = {
            "or_opt": self.orOpt,
            "or_opt_rand": self.orOptRand,
            "or_opt_restrict": self.orOptRestrict
        }
        return table.get(method, self._abstract)

    def _abstract(self, *a, **kw):
        raise NameError(f"Missing method '{self.__method}' in OrOpt")

    def setParameters(self, **kw):
        super().setParameters(**kw)
        if self.__method == "or_opt_restrict" and self._candidates is None:
            k = self.__configs.get("k_candidates", 15)
            self._build_candidate_lists(k)

    def __call__(self, p1, p2, route, **call_configs):
        cfg = dict(self.__configs)
        cfg.update(call_configs)
        return self.__fn(p1, p2, route, **cfg)

    # ---------- helpers ----------
    def _build_candidate_lists(self, k=15):
        n = self.GENOME_LENGTH
        D = self.distance
        cand = []
        for c in range(n):
            nn = np.argsort(D[c])[1:k+1]  # skip self at 0
            cand.append(nn.astype(np.int32))
        self._candidates = cand

    def _route_length(self, r):
        return self.distance[r[-1], r[0]] + np.sum(self.distance[r[:-1], r[1:]])

    def _delta_relocate(self, r, i, L, j):
        """
        Fast delta for relocating block r[i:i+L] to after position j.
        i = block start, L = block length, j = insert-after index in the
        route with the block removed.

        Returns delta (new_length - old_length).
        """
        n = len(r)

        # indices in original route
        a_idx = (i - 1) % n
        b_idx = (i + L) % n
        a = r[a_idx]
        s = r[i]
        e = r[(i + L - 1) % n]
        b = r[b_idx]

        # build reduced route indices to interpret j
        mask = np.ones(n, dtype=bool)
        block_idx = [(i + t) % n for t in range(L)]
        mask[block_idx] = False
        r_red = r[mask]

        m = len(r_red)
        j = j % m
        c = r_red[j]
        d = r_red[(j + 1) % m]

        # removed edges: (a-s), (e-b), (c-d)
        # added edges:   (a-b), (c-s), (e-d)
        old = self.distance[a, s] + self.distance[e, b] + self.distance[c, d]
        new = self.distance[a, b] + self.distance[c, s] + self.distance[e, d]

        return new - old

    def _apply_relocate(self, r, i, L, j):
        """Actually relocate the block."""
        n = len(r)
        block_idx = [(i + t) % n for t in range(L)]
        block = r[block_idx]

        mask = np.ones(n, dtype=bool)
        mask[block_idx] = False
        r_red = r[mask]

        j = j % len(r_red)
        # insert after j -> between j and j+1
        out = np.concatenate([r_red[:j+1], block, r_red[j+1:]])
        return out

    # ---------- methods ----------
    def orOpt(self, p1, p2, route, block_max=3, iters=1, seed=None):
        if seed is not None:
            np.random.seed(seed)

        cur = route.copy()
        cur_len = self._route_length(cur)
        n = len(cur)

        for _ in range(iters):
            best_delta = 0.0
            best_move = None

            for L in range(1, block_max + 1):
                for i in range(n):
                    # reduced route length
                    m = n - L
                    for j in range(m):
                        # skip reinserting into the same place
                        if j == (i - 1) % m:
                            continue
                        delta = self._delta_relocate(cur, i, L, j)
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (i, L, j)

            if best_move is None:
                break

            i, L, j = best_move
            cur = self._apply_relocate(cur, i, L, j)
            cur_len += best_delta

        return cur

    def orOptRand(self, p1, p2, route, block_max=3, iters=1, trials=80, seed=None):
        if seed is not None:
            np.random.seed(seed)
        cur = route.copy()
        cur_len = self._route_length(cur)
        n = len(cur)

        for _ in range(iters):
            improved = False
            for _t in range(trials):
                L = np.random.randint(1, block_max + 1)
                i = np.random.randint(0, n)
                j = np.random.randint(0, n - L)

                delta = self._delta_relocate(cur, i, L, j)
                if delta < 0:
                    cur = self._apply_relocate(cur, i, L, j)
                    cur_len += delta
                    improved = True
                    break
            if not improved:
                break

        return cur

    def orOptRestrict(self, p1, p2, route, block_max=3, iters=1,
                      k_candidates=15, sample_per_iter=60, seed=None):
        if seed is not None:
            np.random.seed(seed)
        if self._candidates is None:
            self._build_candidate_lists(k_candidates)

        cur = route.copy()
        cur_len = self._route_length(cur)
        n = len(cur)

        pos = np.empty(n, dtype=np.int32)
        pos[cur] = np.arange(n)

        for _ in range(iters):
            best_delta = 0.0
            best_move = None

            i_samples = np.random.choice(n, size=min(sample_per_iter, n), replace=False)
            for L in range(1, block_max + 1):
                for i in i_samples:
                    s = cur[i]
                    # consider insertion positions near nearest neighbors of s
                    cand_cities = self._candidates[s]
                    for c in cand_cities:
                        j_pos = pos[c]  # insert near that position in reduced tour
                        m = n - L
                        j = j_pos % m
                        delta = self._delta_relocate(cur, i, L, j)
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (i, L, j)

            if best_move is None:
                break

            i, L, j = best_move
            cur = self._apply_relocate(cur, i, L, j)
            cur_len += best_delta
            pos[cur] = np.arange(n)

        return cur
