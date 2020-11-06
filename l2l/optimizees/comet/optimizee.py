from collections import namedtuple
import warnings
import numpy as np
import sys
import pandas as pd
import os
from os.path import join, isdir, dirname, isfile
from l2l.optimizees.optimizee import Optimizee

CometOptimizeeParameters = \
    namedtuple('CometOptimizeeParameters',
               ['seed',
                'keys_to_evolve',
                'default_params_dict',
                'default_bounds_dict',
                'model_class',
                'experiment_class',
                'test_class'])


class CometOptimizee(Optimizee):
    """
    Implements a simple function optimizee.
    Functions are generated using the FunctionGenerator.
    NOTE: Make sure the optimizee_fitness_weights is set to (-1,)
    to minimize the value of the function

    :param traj:
        The trajectory used to conduct the optimization.

    :param parameters:
        Instance of :func:`~collections.namedtuple`
        :class:`.CometOptimizeeParameters`

    """

    def __init__(self, traj, parameters):
        super().__init__(traj)

        self.seed = parameters.seed
        self.keys_to_evolve = parameters.keys_to_evolve
        self.default_dict = dict()
        self.bounds_dict = dict()
        self.shape_dict = dict()
        for key in parameters.keys_to_evolve:
            self.default_dict[key] = parameters.default_params_dict[key]
            self.bounds_dict[key] = parameters.default_bounds_dict[key]
            self.shape_dict[key] = np.array(self.default_dict[key]).shape

        self.model_class = parameters.model_class
        self.test_class = parameters.test_class

    def create_individual(self):
        """
        Creates a random value of parameter within given bounds

        This function checks, which parameters should be evolved and draws new
        values for them.
        Also ensures that the new values are within predefined bounds.
        """
        parameter_dict = {}
        for key in self.keys_to_evolve:
            n = np.size(self.default_dict[key])
            minval = self.bounds_dict[key]['min']
            maxval = self.bounds_dict[key]['max']
            parameter_dict[key] = np.random.uniform(minval, maxval, size=n)

        return parameter_dict

    def bounding_func(self, individual):
        """
        Bounds the individual via coordinate clipping
        Controls that the new values fall within predefined bounds.
        The entries to the dictionary should be numpy arrays.
        """
        for key in self.keys_to_evolve:
            n = np.size(individual[key])
            try:
                if n == 1:
                    if individual[key] > self.bounds_dict[key]['max']:
                        individual[key] = self.bounds_dict[key]['max']
                    elif individual[key] < self.bounds_dict[key]['min']:
                        individual[key] = self.bounds_dict[key]['min']
                else:
                    if isinstance(individual[key], np.ndarray):
                        ind = individual[key]

                        # Check maximum bounds
                        max_bound = self.bounds_dict[key]['max']
                        max_mask = ind > max_bound
                        ind[max_mask] = max_bound[max_mask]

                        # Check minimum bounds
                        min_bound = self.bounds_dict[key]['min']
                        min_mask = ind < min_bound
                        ind[min_mask] = min_bound[min_mask]

                        individual[key] = ind

                    else:
                        raise TypeError("Expected a numpy array,"
                                        "but {key} is {type(individual[key])}")
            except KeyError:
                print('Undefined bounds for {}'.format[key])

        return individual

    def simulate(self, traj):
        """
        Returns the value of the function chosen during initialization

        :param ~l2l.utils.trajectory.Trajectory traj: Trajectory
        :return: a single element :obj:`tuple` containing the value of
        the chosen function
        """
        # Prevent huge ammounts of expected warnings from being raised
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        # Initialize target model
        if not hasattr(self, 'experiment_class'):
            # Test against synthetic data
            target = self.model_class(name='Synthetic target',
                                      run_params={'seed': self.seed})
        else:
            # Test against experimental data
            # TODO make this line area sensitive
            raise NotImplementedError('Work in progress')
            target = self.experiment_class(name='Experimental target')

        # Initialize observation model
        # The dictionary `model_params` can be used to set some model params
        # The defaul parameters are at comet.models.brunel.model_params
        new_params = {}
        for k in traj.individual.params.keys():
            flat_params = traj.individual.params[k]
            # Recovers the original size of the paramter array
            key = k.split('individual.')[-1]
            new_params[key] = flat_params.reshape(self.shape_dict[key])
        observation = self.model_class(name='Optimizee',
                                       model_params=new_params,
                                       run_params={'seed': self.seed})

        # Initialize the joint test:
        # * Defines which statistics are calculated
        # * Defines which distance metric will be used to calculate the scores
        test = self.test_class()

        # If default predictions exists load them instead of re-calculating
        modulefile = sys.modules[self.model_class.__module__].__file__
        target_pred_path = join(dirname(modulefile),
                                'predictions', 'default.csv')
        if isfile(target_pred_path):
            target_prediction = pd.read_csv(target_pred_path).to_numpy().T
            test.set_prediction(model=target, prediction=target_prediction)
            print('Precalculated default model predictions stored in: ',
                  'memory ' if target._backend.use_memory_cache else '',
                  'disk' if target._backend.use_disk_cache else '')
        else:
            # Target predictions do not exist, therefore they are calculated
            # Should only happen once in the whole execution (if at all)
            print('(Re-)calculating the default model predictions.')
            target_prediction = test.generate_prediction(target)
            df = pd.DataFrame(data=target_prediction.T,
                              columns=[t.name for t in test.test_list],
                              index=np.arange(target_prediction.shape[1]))
            # Store calculated predictions for the default model
            if not isdir(dirname(target_pred_path)):
                os.mkdir(dirname(target_pred_path))
            df.to_csv(target_pred_path, index=False)

        # Run test:
        # * This will run the simulation (for the observation model)
        # * Estimate the statistics and save them as csv files
        # * Then the Wasserstein distance is calculated (for each pair)
        score = test.judge([target, observation],
                           only_lower_triangle=True).iloc[1, 0]
        score = [score.score]

        return (score)
