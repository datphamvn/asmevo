"""Evolution nodes: parent selection and survivor selection around the shared ``Mutate``.

``shared["population"]`` holds the -O3 baseline and up to ``population_size - 1`` of the
fittest distinct valid candidates so far.
"""

from __future__ import annotations

from pocketflow import Node

from asmevo.search.selection import survive, tournament


class Select(Node):
    """Start the next generation: one tournament winner of the population per offspring."""

    def prep(self, shared):
        return shared["population"], shared["cfg"].search, shared["rng"]

    def exec(self, prep_res):
        population, cfg, rng = prep_res
        return [tournament(population, cfg.tournament_size, rng)
                for _ in range(cfg.offspring_per_generation)]

    def post(self, shared, prep_res, exec_res):
        shared["generation"] += 1
        shared["parents"] = exec_res
        shared["offspring"] = []


class Survive(Node):
    """Replace the population with the baseline and the fittest of population + offspring.

    Returns ``"select"`` until ``generations`` generations were made, then ``"done"``.
    """

    def prep(self, shared):
        return shared["population"], tuple(shared["offspring"]), shared["cfg"].search

    def exec(self, prep_res):
        population, offspring, cfg = prep_res
        return survive(population, offspring, cfg.population_size)

    def post(self, shared, prep_res, exec_res):
        shared["population"] = exec_res
        return "select" if shared["generation"] < prep_res[2].generations else "done"
