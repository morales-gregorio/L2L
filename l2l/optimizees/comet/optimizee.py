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
               ['keys_to_evolve',
                'default_params_dict',
                'default_bounds_dict',
                'simulation_params',
                'model_class',
                'target_class',
                'target_predictions_csv',
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

        self.keys_to_evolve = parameters.keys_to_evolve
        self.default_params = parameters.default_params_dict
        self.run_params = parameters.simulation_params
        self.bounds_dict = dict()
        self.shape_dict = dict()

        for key in parameters.keys_to_evolve:
            self.bounds_dict[key] = parameters.default_bounds_dict[key]
            self.shape_dict[key] = np.array(self.default_params[key]).shape

        self.model_class = parameters.model_class
        self.target_class = parameters.target_class
        self.target_pred_csv = parameters.target_predictions_csv
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

            # Draw random samples within the parameter bounds
            minval = self.bounds_dict[key]['min']
            maxval = self.bounds_dict[key]['max']
            params = np.random.uniform(minval, maxval)
            # Individuals always have flattened lists instead of arrays
            # the array shape is separately stored and used later to reshape
            parameter_dict[key] = params.flatten()

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
                        max_bound = self.bounds_dict[key]['max'].flatten()
                        max_mask = ind > max_bound
                        ind[max_mask] = max_bound[max_mask]

                        # Check minimum bounds
                        min_bound = self.bounds_dict[key]['min'].flatten()
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

        # Initialize observation model
        # The dictionary `model_params` can be used to set some model params
        # The defaul parameters are at comet.models.brunel.model_params
        new_params = {}
        for k in traj.individual.params.keys():
            flat_params = traj.individual.params[k]
            # Recovers the original size of the paramter array
            key = k.split('individual.')[-1]
            new_params[key] = flat_params.reshape(self.shape_dict[key])

        model_params = self.default_params.copy()
        model_params.update(new_params)
        observation = self.model_class(name='Optimizee',
                                       model_params=model_params,
                                       run_params=self.run_params)

        # Instantiate test
        test = self.test_class()

        # Instantiate target and set pre-calculated predictions
        target = self.target_class(name='Target')
        target_prediction = pd.read_csv(self.target_pred_csv).to_numpy().T
        test.set_prediction(model=target, prediction=target_prediction)

        # Run test:
        # * This will run the simulation (for the observation model)
        # * Estimate the statistics and save them as csv files
        # * Then the Wasserstein distance is calculated (for each pair)
        score = test.judge([target, observation],
                           only_lower_triangle=True).iloc[1, 0]

        # The format in which the score is returned is important
        score = [score.score]
        return (score)
