from neat.reporting import BaseReporter
import logging
import time

from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys

log = logging.getLogger(__name__)


class LoggingReporter(BaseReporter):
    """Uses logging to output information about the run; an example reporter class."""
    def __init__(self, show_species_detail):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0

    def start_generation(self, generation):
        self.generation = generation
        log.info('****** Running generation %d ******', generation)
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        if self.show_species_detail:
            log.info('Population of %d members in %d species:', ng, ns)
            sids = list(iterkeys(species_set.species))
            sids.sort()
            log.info("   ID   age  size  fitness  adj fit  stag")
            log.info("  ====  ===  ====  =======  =======  ====")
            for sid in sids:
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
                af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                st = self.generation - s.last_improved
                log.info(
                    "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}".format(sid, a, n, f, af, st))
        else:
            log.info('Population of %d members in %d species:', ng, ns)

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        log.info('Total extinctions: %d', self.num_extinctions)
        if len(self.generation_times) > 1:
            log.info("Generation time: %f sec (%f average)", elapsed, average)
        else:
            log.info("Generation time: %f sec", elapsed)

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        log.info('Population\'s average fitness: %f stdev: %f', fit_mean, fit_std)
        log.info(
            'Best fitness: %f - size: %d - species %s - id %s', best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key)

    def complete_extinction(self):
        self.num_extinctions += 1
        log.error('All species extinct!')

    def found_solution(self, config, generation, best):
        log.info('\nBest individual in generation %d meets fitness threshold of %f - complexity: %s',
            self.generation, config.fitness_threshold, best.size())

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            log.debug("Species %s with %d members is stagnated: removing it", sid, len(species.members))

    def info(self, msg):
        log.info(msg)