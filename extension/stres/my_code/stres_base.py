#!/usr/bin/python

from extension.ga_base import GABase


class StresBase(GABase):
    """
    Minimal / no-op stress operator.

    Used when we want to disable stress logic but the GA still expects a
    'stres' object. Calling this object will effectively do nothing.
    """

    def __init__(self, method="noop", name="StresBase", **configs):
        # GABase needs a 'method' string; we just use 'noop' as a label.
        super().__init__(method=method, name=name, **configs)

    def __call__(self, genoms, scores, *args, **kwargs):
        """
        No-op: do not modify the population or scores.

        GeneticAlgorithmManager.my_code.stres.Stres ignores the return value,
        so simply returning (genoms, scores) is safe and harmless.
        """
        return genoms, scores

    # Optional: if something ever calls .noop() explicitly, keep it safe:
    def noop(self, genoms, scores, *args, **kwargs):
        return genoms, scores
