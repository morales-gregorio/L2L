import random
import numpy as np
import pandas as pd

from collections import namedtuple
from copy import deepcopy
from itertools import repeat

from l2l import dict_to_list, list_to_dict
from l2l.optimizers.optimizer import Optimizer

from sklearn.neighbors import NearestNeighbors

RandomSearchParameters = namedtuple('RandomSearchParameters',
                                    ['seed', 'pop_size', 'n_iteration',
                                     'mut_sigma', 'p_survival',
                                     'p_from_best', 'n_best', 'p_gradient',
                                     'ind_list_path'])
RandomSearchParameters.__doc__ = """
:param seed: Random seed
:param pop_size: Size of the population
:param n_iteration: Number of generations simulation should run for
:param mut_sigma: Standard deviation for the gaussian addition mutation.
:param p_survival: Percentage of the population that will not be discarded
  before the next generation.
:param p_from_best: Percentage of the population that will be mutations of the
  best individuals in the entire optimization.
:param n_best: number of individuals to keep in the list of best individuals.
:param p_gradient: probability of a mutation following the gradient descent
  direction, otherwise a random direction is chosen.
:param ind_list_path: path to csv file containing all the explored individuals
"""


class simpleIndividual(list):
    def __init__(self, *args):
        list.__init__(self, *args)
        self.fitness = None


def _mutGaussian(individual, mu, sigma, gradient=None):
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

    # Estimate perturbation
    mag = random.gauss(mu, sigma)
    if gradient is None:
        # Create random vector
        vec = np.array([random.random() for _ in range(size)])
        vec = mag * vec / np.sqrt(np.sum(vec**2))
    else:
        # Use gradient direction
        vec = mag * gradient

    # Mutate individual
    mutant = deepcopy(individual)
    for i in range(size):
        mutant[i] += vec[i]

    return mutant


class RandomSearchOptimizer(Optimizer):
    """
    Implements evolutionary algorithm

    :param  ~l2l.utils.trajectory.Trajectory traj: Use this trajectory to store
      the parameters of the specific runs.
      The parameters should be initialized based on the values in `parameters`
    :param optimizee_create_individual: Function that creates a new individual
    :param optimizee_fitness_weight: Fitness weight. Used to determine if it is
      a maximization or minimization problem. NOTE: this just takes an integer
      value for now!
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
        traj.f_add_parameter('p_from_best', parameters.p_from_best,
                             comment='Portion of generation sampled from best')
        traj.f_add_parameter('n_best', parameters.n_best,
                             comment='Length of list of best individuals')
        traj.f_add_parameter('p_gradient', parameters.p_gradient,
                             comment='Probability of using gradient')
        traj.f_add_parameter('ind_list_path', parameters.ind_list_path,
                             comment='Path to global indvidual list')

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
        self.best_individuals = None
        self.best_ind_fitnesses = None

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

        gen_fitnesses = np.array(gen_fitnesses).flatten()
        traj.v_idx = -1  # set the trajectory back to default

        print("-- End of generation {} --".format(self.g))
        best_ind = self.pop[np.argmax(gen_fitnesses*self.weight)]
        self.best_individual = list_to_dict(best_ind, self.ind_dict_spec)

        # Save the best individuals into a list
        if self.best_individuals is None:
            b_and_g = self.pop
            bng_fitnesses = gen_fitnesses
        else:
            b_and_g = self.best_individuals + self.pop
            bng_fitnesses = np.concatenate([self.best_ind_fitnesses,
                                           gen_fitnesses])

        best_idx = np.argsort(self.weight*bng_fitnesses)[-traj.n_best:]
        best_idx = best_idx.astype(int)
        sorting = np.argsort((-self.weight*bng_fitnesses[best_idx]))
        sorting = sorting.astype(int)
        self.best_individuals = [b_and_g[idx] for idx in best_idx[sorting]]
        self.best_ind_fitnesses = bng_fitnesses[best_idx][sorting]

        print('\nOverall best individuals are:')
        for ind in self.best_individuals:
            ind_dict = list_to_dict(ind, self.ind_dict_spec)
            print("\t%s, %s" % (ind_dict, ind.fitness))

        print('\nBest individual from generation is:')
        print("\t%s, %s" % (self.best_individual, best_ind.fitness))

        print('\n--Saving individuals list--\n')
        # Compile results into a dataframe
        result = traj.results['all_results']
        ind_results = []
        for gen in traj.individuals.keys():
            if result[gen]:
                for i, ind in enumerate(traj.individuals[gen]):
                    res = [gen, ind.ind_idx]
                    for param in ind.params.keys():
                        res += list(ind.params[param])
                    res += result[gen][i][1]  # WS distance
                    ind_results.append(res)
        # Resolve the parameter labels
        labels = ['Generation', 'Individual']
        param_labels = []
        for key in ind.params.keys():
            vals = ind.params[param]
            if np.size(vals) == 1:
                param_labels.append(key.split('individual.')[-1])
            else:
                newlabels = [key.split('individual.')[-1] + '_' + str(i)
                             for i in range(len(list(vals)))]
                param_labels += newlabels
        labels += param_labels
        labels.append('Score')
        df = pd.DataFrame(ind_results, columns=labels)
        df.to_csv(traj.ind_list_path, index=False)

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

            # Select some individuals out of the best
            m = int(traj.pop_size * traj.p_from_best)
            some_of_the_best_idx = np.random.choice(range(traj.n_best), m)
            some_of_the_best = [self.best_individuals[idx]
                                for idx in some_of_the_best_idx]
            survivors = survivors + some_of_the_best

            print('\Selected best individuals are:')
            for ind in some_of_the_best:
                ind_dict = list_to_dict(ind, self.ind_dict_spec)
                print("\t%s, %s" % (ind_dict, ind.fitness))

            # Estimate gradient for the survivors
            gen_mask, ind_mask = [], []
            for ind in survivors:
                close_params = np.isclose(df[param_labels], ind)
                ind_mask.append(np.all(close_params, axis=1))
                gen_mask.append(df['Generation'] == self.g)
            gen_mask, ind_mask = np.array(gen_mask), np.array(ind_mask)
            mask = np.any(gen_mask, axis=0) & np.any(ind_mask, axis=0)
            gradient = self.natural_gradient(df, param_labels, mask=mask)

            print('\nGradients are:')
            print(gradient)

            # Mutate all the survivors, sometimes using the gradient
            offspring = []
            for j, ind in enumerate(survivors):
                print(j)
                if random.random() < traj.p_gradient:
                    print('using gradient')
                    mutant = _mutGaussian(ind, mu=0, sigma=traj.mut_sigma,
                                          gradient=gradient[j])
                else:
                    print('Random walk')
                    mutant = _mutGaussian(ind, mu=0, sigma=traj.mut_sigma)
                del mutant.fitness
                mutant = self.optimizee_bounding_func(
                    list_to_dict(mutant, self.ind_dict_spec))
                offspring.append(simpleIndividual(dict_to_list(mutant)))

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

    def natural_gradient(self, df, param_labels, mask=None, N=10):
        # Get nearest neighbours
        X = df[param_labels]
        nbrs = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        if mask is not None:
            distances, indices = distances[mask], indices[mask]

        # Estimate gradient at each mask point
        gradients = []
        for i in range(len(indices)):
            # Get a ball
            ball = df.iloc[indices[i]]
            dist = distances[i]
            ball_X = ball[param_labels]

            # Estimate the gradient at that point
            ball_d = (ball_X.iloc[1:] - ball_X.iloc[0]).divide(dist[1:],
                                                               axis='rows')
            w = ball['Score'].iloc[1:] - ball['Score'].iloc[0]
            g = ball_d.multiply(w, axis='rows').sum(axis='rows') / N
            gradients.append(g)

        G = np.stack(gradients)

        # Normalization and direction
        nG = self.weight * G / np.sqrt(np.sum(G**2, axis=1))[:, None]

        return nG

    def end(self, traj):
        """
        See :meth:`~l2l.optimizers.optimizer.Optimizer.end`
        """
        # ------------ Finished all runs and print result --------------- #
        print("-- End of (successful) evolution --")
