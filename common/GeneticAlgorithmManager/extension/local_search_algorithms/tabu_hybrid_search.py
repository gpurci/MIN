import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import RootGA


class TabuHybridSearch(RootGA):
    """
    Hybrid tabu for TSP/route part of TTP.

    Supported method:
        - "hybrid_2opt"

    Configs (defaults are reasonable for a280):
        iters: 100                 # tabu iterations per call
        k_candidates: 15           # nearest neighbors per city
        tenure: 7                  # tabu tenure (iterations)
        sample_per_iter: 60        # max candidate moves evaluated per iter
        seed: None                 # rng seed for reproducibility
    """
    def __init__(self, method="hybrid_2opt", dataset=None, **configs):
        super().__init__()
        if dataset is None:
            raise ValueError("TabuHybridSearch requires dataset=<dict> with 'distance'.")
        self.dataset = dataset
        self.distance = dataset["distance"]
        self.__method = method
        self.__configs = configs
        self.__fn = self.__unpack(method)

        # candidate lists are computed lazily after GENOME_LENGTH is set
        self._candidates = None

    def __str__(self):
        return f"TabuHybridSearch(method={self.__method}, configs={self.__configs})"

    def help(self):
        print("""TabuHybridSearch:
    method: 'hybrid_2opt'
    configs:
        iters=100
        k_candidates=15
        tenure=7
        sample_per_iter=60
        seed=None
""")

    def __unpack(self, method):
        table = {
            "hybrid_2opt": self.hybrid2opt
        }
        return table.get(method, self._abstract)

    def _abstract(self, *a, **kw):
        raise NameError(f"Missing method '{self.__method}' in TabuHybridSearch")

    def setParameters(self, **kw):
        """
        Called by GA. We use it to build candidate lists once GENOME_LENGTH exists.
        """
        super().setParameters(**kw)
        if self.GENOME_LENGTH and self._candidates is None:
            self._build_candidate_lists(
                #Build nearest-neighbor candidate lists.
                k=self.__configs.get("k_candidates", 15)
            )

    def __call__(self, parent1, parent2, offspring, **call_configs):
        cfg = dict(self.__configs)
        cfg.update(call_configs)
        return self.__fn(parent1, parent2, offspring, **cfg)


    # Candidate list construction (k-nearest neighbors per city)
    def _build_candidate_lists(self, k=15):
        n = self.GENOME_LENGTH
        D = self.distance
        cand = []
        #Loop over each city.
        for c in range(n):
            # Skip city 0
            nn = np.argsort(D[c])[1:k+1]
            cand.append(nn.astype(np.int32))
        self._candidates = cand

    # Hybrid Tabu with restricted 2-opt neighborhood
    def hybrid2opt(self, parent1, parent2, route,
                   iters=100, tenure=7, sample_per_iter=60,
                   k_candidates=15, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if self._candidates is None:
            self._build_candidate_lists(k_candidates)

        n = len(route)
        pos = np.empty(n, dtype=np.int32)
        pos[route] = np.arange(n, dtype=np.int32)

        def dist(a, b):
            return self.distance[a, b]
        #Computes total TSP length:
        def route_length(r):
            return dist(r[-1], r[0]) + np.sum(self.distance[r[:-1], r[1:]])

        cur = route.copy()
        cur_len = route_length(cur)
        best = cur.copy()
        best_len = cur_len

        # tabu list over removed edges (unordered city pairs)
        # store as dict: edge -> expire_iter
        tabu = {}

        #Ensures an undirected edge is stored consistently.
        def edge_key(u, v):
            return (u, v) if u < v else (v, u)

        for t in range(iters):
            # purge expired tabu (they are no longer tabu, you can use them again)
            if tabu:
                expired = [e for e, exp in tabu.items() if exp <= t]
                for e in expired:
                    tabu.pop(e, None)

            # Build a pool of promising 2-opt moves using candidate list.
            moves = []
            # sample a few indices i to keep cost bounded
            # choose i in [0..n-1]
            i_samples = np.random.choice(n, size=min(sample_per_iter, n), replace=False)
            for i in i_samples:
                a = cur[i]
                b = cur[(i+1) % n]
                # consider neighbors of a (or b) as possible c
                for c in self._candidates[a]:
                    j = pos[c]
                    if j == i or (j+1) % n == i:
                        continue
                    d = cur[(j+1) % n]

                    # 2-opt removes edges (a-b) and (c-d), adds (a-c) and (b-d)
                    removed1 = edge_key(a, b)
                    removed2 = edge_key(c, d)
                    added1   = edge_key(a, c)
                    added2   = edge_key(b, d)

                    # compute delta quickly (delta>0 -> good move or else bad)
                    delta = (dist(a, c) + dist(b, d)) - (dist(a, b) + dist(c, d))
                    moves.append((delta, i, j, removed1, removed2, added1, added2))

            if not moves:
                break

            # choose best admissible move
            moves.sort(key=lambda x: x[0])
            chosen = None
            for delta, i, j, rem1, rem2, add1, add2 in moves:
                new_len = cur_len + delta

                is_tabu = (add1 in tabu) or (add2 in tabu)
                # aspiration if improves best found in this tabu run
                if (not is_tabu) or (new_len < best_len):
                    chosen = (delta, i, j, rem1, rem2, new_len)
                    break

            if chosen is None:
                # all moves tabu and none aspirational -> stop
                break

            delta, i, j, rem1, rem2, new_len = chosen

            # apply 2-opt: reverse segment between i+1 and j inclusive
            # ensure i < j in index space for reversal
            if i < j:
                cur[i+1:j+1] = cur[i+1:j+1][::-1]
                # update positions only for affected segment
                seg = cur[i+1:j+1]
                pos[seg] = np.arange(i+1, j+1, dtype=np.int32)
            else:
                # wrap-around reversal: do it by rebuilding
                idx = np.arange(n)
                # segment after i and up to j (wrapped)
                seg_idx = np.concatenate((idx[i+1:], idx[:j+1]))
                seg = cur[seg_idx][::-1]
                cur[seg_idx] = seg
                pos[seg] = seg_idx

            cur_len = new_len

            # update tabu with removed edges for 'tenure' iterations
            tabu[rem1] = t + tenure
            tabu[rem2] = t + tenure

            if cur_len < best_len:
                best_len = cur_len
                best = cur.copy()

        return best