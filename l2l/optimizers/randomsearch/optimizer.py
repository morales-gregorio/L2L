import random
import numpy as np

from collections import namedtuple
from copy import deepcopy
from itertools import repeat

from l2l import dict_to_list, list_to_dict
from l2l.optimizers.optimizer import Optimizer

RandomSearchParameters = namedtuple('GeneticAlgorithmParameters',
                                    ['seed', 'pop_size', 'n_iteration',
                                     'mut_sigma', 'p_survival'])
RandomSearchParameters.__doc__ = """
:param seed: Random seed
:param pop_size: Size of the population
:param n_iteration: Number of generations simulation should run for
:param mut_sigma: Standard deviation for the gaussian addition mutation.
:param p_survival: Percentage of the population that will not be discarded
  before the next generation.
"""


class simpleIndividual(list):
    def __init__(self, *args):
        list.__init__(self, *args)
        self.fitness = None


def _mutGaussian(individual, mu, sigma):
    """This function applies a gaussian mutation of mean *mu* and standard
    deviation *sigma* on the input individual. This mutation expects a
    :term:`sequence` individual composed of real valued attributes.

    :param individual: Individual to be mutated.
    :param mu: Mean for the gaussian addition mutation.
    :param sigma: Standard deviation for the gaussian addition mutation.
    :returns: A mutated copy of the input individual

    This function uses the :func:`~random.random` and :func:`~random.gauss`
    functions from the python base :mod:`random` module.
    """
    size = len(individual)
    mu = repeat(mu, size)
    sigma = repeat(sigma, size)

    mutant = deepcopy(individual)
    for i, m, s in zip(range(size), mu, sigma):
        mutant[i] += random.gauss(m, s)

    return mutant


class RandomSearchOptimizer(Optimizer):
    """
    Implements evolutionary algorithm

    :param  ~l2l.utils.trajectory.Trajectory traj: Use this trajectory to store
      the parameters of the specific runs.
      The parameters should be initialized based on the values in `parameters`
    :param optimizee_create_individual: Function that creates a new individual
    :param optimizee_fitness_weight: Fitness weight. Used to determine if it is
      a maximization or minimization problem.
    :param parameters: Instance of :func:`~collections.namedtuple`
      :class:`.RandomSearchOptimizer` containing the parameters
      needed by the Optimizer
    """

    def __init__(self, traj,
                 optimizee_create_individual,
                 optimizee_fitness_weight,
                 parameters,
                 optimizee_bounding_func=None):

        self.new_ind = optimizee_create_individual
        self.weight = optimizee_fitness_weight

        super().__init__(traj,
                         optimizee_create_individual=self.new_ind,
                         optimizee_fitness_weights=optimizee_fitness_weight,
                         parameters=parameters,
                         optimizee_bounding_func=optimizee_bounding_func)
        self.optimizee_bounding_func = optimizee_bounding_func
        __, self.ind_dict_spec = \
            dict_to_list(self.new_ind(), get_dict_spec=True)

        traj.f_add_parameter('seed', parameters.seed, comment='Seed for RNG')
        traj.f_add_parameter('pop_size', parameters.pop_size,
                             comment='Population size')
        traj.f_add_parameter('n_iteration', parameters.n_iteration,
                             comment='Number of generations')
        traj.f_add_parameter('mut_sigma', parameters.mut_sigma,
                             comment='Standard deviation for mutation')
        traj.f_add_parameter('p_survival', parameters.p_survival,
                             comment='Survivor percentage in each generation')

        # ------- Initialize Population and Trajectory -------- #
        # NOTE: The Individual object implements the list interface.
        self.pop = []
        for _ in range(parameters.pop_size):
            ind = simpleIndividual(dict_to_list(self.new_ind()))
            self.pop.append(ind)
        self.eval_pop_inds = [ind for ind in self.pop if not ind.fitness]
        self.eval_pop = [list_to_dict(ind, self.ind_dict_spec)
                         for ind in self.eval_pop_inds]

        self.g = 0  # current generation
        self.best_individual = None

        self._expand_trajectory(traj)

    def post_process(self, traj, fitnesses_results):
        """
        See :meth:`~l2l.optimizers.optimizer.Optimizer.post_process`
        """
        NGEN = traj.n_iteration

        print("  Evaluating %i individuals" % len(fitnesses_results))

        # *******************************************************************
        # Storing run-information in the trajectory
        # Reading fitnesses and performing distribution update
        # *******************************************************************
        gen_fitnesses = []
        for run_index, fitness in fitnesses_results:
            # We need to convert the current run index into an ind_idx
            # (index of individual within one generation)
            traj.v_idx = run_index
            ind_index = traj.par.ind_idx

            traj.f_add_result('$set.$.individual', self.eval_pop[ind_index])
            traj.f_add_result('$set.$.fitness', fitness)

            # Use the ind_idx to update the fitness
            individual = self.eval_pop_inds[ind_index]
            individual.fitness = fitness
            gen_fitnesses.append(fitness)

        gen_fitnesses = np.array(gen_fitnesses)
        traj.v_idx = -1  # set the trajectory back to default

        print("-- End of generation {} --".format(self.g))
        best_ind = self.pop[np.argmax(gen_fitnesses*self.weight)]
        self.best_individual = list_to_dict(best_ind, self.ind_dict_spec)
        print('Best individual is:')
        print("\t%s, %s" % (self.best_individual, best_ind.fitness))

        # --Create the next generation by discarding the worst individuals -- #
        if self.g < NGEN - 1:  # not necessary for the last generation
            # Select the best individuals from the current generation
            percentiles = np.argsort(gen_fitnesses) / len(gen_fitnesses)
            if self.weight > 0:
                # Maximization case
                survivor_mask = percentiles > traj.p_survival
            elif self.weight < 0:
                # Minimization case
                survivor_mask = percentiles < traj.p_survival
            survivors = [ind for ind, survived in zip(self.pop, survivor_mask)
                         if survived]

            print('\nSurvivors are:')
            for ind in survivors:
                ind_dict = list_to_dict(ind, self.ind_dict_spec)
                print("\t%s, %s" % (ind_dict, ind.fitness))

            # Mutate all the survivors
            offspring = []
            for ind in survivors:
                mutant = _mutGaussian(ind, mu=0, sigma=traj.mut_sigma)
                del mutant.fitness
                mutant = self.optimizee_bounding_func(
                    list_to_dict(mutant, self.ind_dict_spec))
                offspring.append(dict_to_list(mutant))

            # Create new random individuals to replace the dead ones
            for _ in range(len(self.pop) - len(survivors)):
                ind = simpleIndividual(dict_to_list(self.new_ind()))
                offspring.append(ind)

            # The population is entirely replaced by the offspring
            self.pop[:] = offspring

            self.eval_pop_inds = [ind for ind in self.pop if not ind.fitness]
            self.eval_pop = [list_to_dict(ind, self.ind_dict_spec)
                             for ind in self.eval_pop_inds]

            self.g += 1  # Update generation counter
            self._expand_trajectory(traj)

    def end(self, traj):
        """
        See :meth:`~l2l.optimizers.optimizer.Optimizer.end`
        """
        # ------------ Finished all runs and print result --------------- #
        print("-- End of (successful) evolution --")
