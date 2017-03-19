
import logging
from collections import namedtuple, Counter

import numpy as np

from ltl.optimizers.optimizer import Optimizer
from ltl import dict_to_list
from ltl import list_to_dict
from itertools import compress
logger = logging.getLogger("ltl-lsrs")

LineSearchRestartParameters = namedtuple('LineSearchRestartParameters',
                                          ['n_iterations', 'pop_size', 'line_search_iterations', 'bounds_min', 'bounds_max'])
LineSearchRestartParameters.__doc__ = """
:param n_parallel_runs: Number of individuals per simulation / Number of parallel Simulated Annealing runs
:param noisy_step: Size of the random step
:param temp_decay: A function of the form f(t) = temperature at time t
:param n_iteration: number of iteration to perform
:param stop_criterion: Stop if change in fitness is below this value
:param seed: Random seed
"""


class LineSearchRestartOptimizer(Optimizer):
    """
    Class for a gernic line search restart optimizer
    In the pseudo code the algorithm does:

    For n iterations do:
        - Take a step of size noisy step in a random direction
        - If it reduces the cost, keep the solution
        - Otherwise keep with probability exp(- (f_new - f) / T)

    NOTE: This expects all parameters of the system to be of floating point

    :param  ~pypet.trajectory.Trajectory traj:
      Use this pypet trajectory to store the parameters of the specific runs. The parameters should be
      initialized based on the values in `parameters`
    
    :param optimizee_create_individual:
      Function that creates a new individual
    
    :param optimizee_fitness_weights: 
      Fitness weights. The fitness returned by the Optimizee is multiplied by these values (one for each
      element of the fitness vector)
    
    :param parameters: 
      Instance of :func:`~collections.namedtuple` :class:`SimulatedAnnealingParameters` containing the
      parameters needed by the Optimizer
    
    :param optimizee_bounding_func:
      This is a function that takes an individual as argument and returns another individual that is
      within bounds (The bounds are defined by the function itself). If not provided, the individuals
      are not bounded.
    """

    def __init__(self, traj, optimizee_create_individual, optimizee_fitness_weights, parameters,
                 optimizee_bounding_func=None):
        super().__init__(traj, optimizee_create_individual=optimizee_create_individual,
                         optimizee_fitness_weights=optimizee_fitness_weights, parameters=parameters)
        self.optimizee_bounding_func = optimizee_bounding_func
        
        # The following parameters are recorded
        traj.f_add_parameter('n_iterations', parameters.n_iterations,
                             comment='Number of iterations to perform')
        traj.f_add_parameter('pop_size', parameters.pop_size,
                             comment='Number of different starting points to consider')
        traj.f_add_parameter('line_search_iterations', parameters.line_search_iterations,
                             comment='Number of line search iterations')
        

        _, self.optimizee_individual_dict_spec = dict_to_list(self.optimizee_create_individual(), get_dict_spec=True)

        # Note that this array stores individuals as an np.array of floats as opposed to Individual-Dicts
        # This is because this array is used within the context of the simulated annealing algorithm and
        # Thus needs to handle the optimizee individuals as vectors
        self.bounds_min = parameters.bounds_min
        self.bounds_max = parameters.bounds_max
        self.dimensions = self.optimizee_individual_dict_spec[0][2]
        
#         self.current_individual_list = [np.array(dict_to_list(self.optimizee_create_individual()))
#                                         for _ in range(parameters.pop_size)]        

        traj.f_add_result('fitnesses', [], comment='Fitnesses of all individuals')

        # The following parameters are NOT recorded
        self.g = 0  # the current generation
        self.current_line_search_iteration = 0
        self.current_restart_iteration = 0
        self.line_search_variation_value = 1
        self.gradient_evaluated = False

        # Keep track of current fitness value to decide whether we want the next individual to be accepted or not
        self.old_fitness_value_list = [-np.Inf] * parameters.pop_size

        self.eval_pop = self.generateIndividuals(parameters.pop_size, parameters.bounds_min, parameters.bounds_max)
        self.old_eval_pop = self.eval_pop 
        self._expand_trajectory(traj)

    def post_process(self, traj, fitnesses_results):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.post_process`
        """
        n_iterations, pop_size, line_search_iterations =  \
            traj.n_iterations, traj.pop_size, traj.line_search_iterations
            
        #Check if a line search should still be performed
        if self.current_line_search_iteration < line_search_iterations:
            #Perform the line search           
            k = self.current_line_search_iteration + self.current_restart_iteration * line_search_iterations
            ak = 2 + 3 / (2 ** (k ** 2) + 1)
#             pk = -1;
            pk = np.random.uniform(-self.line_search_variation_value, self.line_search_variation_value, (pop_size, self.dimensions))
            update_list = (ak * pk).tolist()
            
            #Compare the points and take the better one
            weighted_fitness_list = []
            for i, (run_index, fitness) in enumerate(fitnesses_results):
                 
                # Update fitnesses
                # NOTE: The fitness here is a tuple! For now, we'll only support fitnesses with one element
                weighted_fitness = sum(f * w for f, w in zip(fitness, self.optimizee_fitness_weights))
                weighted_fitness_list.append(weighted_fitness)
     
            #Compare the points and take the better one
            comparison_result = np.greater(weighted_fitness_list, self.old_fitness_value_list)
            
            #Choose the next points according to their fitness improvement
            temp_pop = []
            temp_fitness = []
            for ind in range(pop_size):
                temp_pop.append(comparison_result[ind] and self.eval_pop[ind] or self.old_eval_pop[ind])
                temp_fitness.append(comparison_result[ind] and weighted_fitness_list[ind] or self.old_fitness_value_list[ind])
            
            self.old_eval_pop = temp_pop
            self.old_fitness_value_list = temp_fitness
            
            #Modify individuals for next run
            for i, (run_index, fitness) in enumerate(fitnesses_results):
            
                # We need to convert the current run index into an ind_idx
                # (index of individual within one generation)
                traj.v_idx = run_index
                ind_index = traj.par.ind_idx
                
                #Update the individual
                individual = np.array(dict_to_list(self.eval_pop[ind_index])) + update_list[ind_index]

                new_individual = list_to_dict(individual, self.optimizee_individual_dict_spec)
                if self.optimizee_bounding_func is not None:
                    new_individual = self.optimizee_bounding_func(new_individual)
                self.eval_pop[ind_index] = new_individual
            
            self.current_line_search_iteration += 1
            
            traj.v_idx = -1  # set the trajectory back to default
            fitnesses_results.clear()
            self._expand_trajectory(traj)
            
        elif self.current_restart_iteration < n_iterations:
            #Perform the restarting procedure        
            
            if self.gradient_evaluated:
                #Reset the line search counter
                self.current_line_search_iteration = 0
                self.current_restart_iteration += 1
                self.gradient_evaluated = False
                
                #Evaluate the gradient of each parameter according to the best individual
                weighted_fitness_list = []
                distances = []
                for i, (run_index, fitness) in enumerate(fitnesses_results):
                     
                    # Update fitnesses
                    # NOTE: The fitness here is a tuple! For now, we'll only support fitnesses with one element
                    weighted_fitness = sum(f * w for f, w in zip(fitness, self.optimizee_fitness_weights))
                    weighted_fitness_list.append(weighted_fitness)
                    distances.append(np.subtract(self.old_eval_pop[0], dict_to_list(self.eval_pop[i])))
                    
                fitness_distances = np.reshape(np.subtract(weighted_fitness_list, self.old_fitness_value_list[0]), (pop_size, 1))
                fitness_distances = np.hstack((fitness_distances, fitness_distances))
                
                gradient = np.mean(np.divide(fitness_distances, distances), axis=0)
                
                for dim in range(self.dimensions):
                    if gradient[dim] > 0:
                        self.bounds_max[dim] = self.old_eval_pop[0][dim]
                    else:
                        self.bounds_min[dim] = self.old_eval_pop[0][dim]
                
                #Generate new samples from new boundaries
                self.eval_pop = self.generateIndividuals(pop_size, self.bounds_min, self.bounds_max)
                self.old_eval_pop = self.eval_pop
                self.old_fitness_value_list = [-np.Inf] * pop_size
                self.gradient_evaluated = False
                traj.v_idx = -1  # set the trajectory back to default
                fitnesses_results.clear()
                self._expand_trajectory(traj)
                
            else:
                
                #Select the best point and the corresponding result
                for i, (run_index, fitness) in enumerate(fitnesses_results):
                      
                    # Update fitnesses
                    # NOTE: The fitness here is a tuple! For now, we'll only support fitnesses with one element
                    weighted_fitness = sum(f * w for f, w in zip(fitness, self.optimizee_fitness_weights))
                    self.old_fitness_value_list.append(weighted_fitness)
                     
                    # We need to convert the current run index into an ind_idx
                    # (index of individual within one generation)
                    traj.v_idx = run_index
                    ind_index = traj.par.ind_idx
                    self.old_eval_pop.append(self.eval_pop[ind_index])
                     
                best_last_indiv_index = np.argmax(self.old_fitness_value_list)
                best_last_indiv = dict_to_list(self.old_eval_pop[best_last_indiv_index])
                best_last_fitness = self.old_fitness_value_list[best_last_indiv_index]
            
                #Prepare the population to evaluate the gradient
                self.old_fitness_value_list = [best_last_fitness]
                self.old_eval_pop = [best_last_indiv]
                
                self.eval_pop = self.generateIndividuals(pop_size, np.subtract(best_last_indiv, 0.01), np.add(best_last_indiv, 0.01))
                self.gradient_evaluated = True
                traj.v_idx = -1  # set the trajectory back to default
                fitnesses_results.clear()
                self._expand_trajectory(traj)
            
            
#         old_eval_pop = self.eval_pop.copy()
#         self.eval_pop.clear()
#         self.T *= temp_decay
# 
#         logger.info("  Evaluating %i individuals" % len(fitnesses_results))
#         # NOTE: Currently works with only one individual at a time.
#         # In principle, can be used with many different individuals evaluated in parallel
#         assert len(fitnesses_results) == traj.n_parallel_runs
#         weighted_fitness_list = []
#         for i, (run_index, fitness) in enumerate(fitnesses_results):
#             
#             # Update fitnesses
#             # NOTE: The fitness here is a tuple! For now, we'll only support fitnesses with one element
#             weighted_fitness = sum(f * w for f, w in zip(fitness, self.optimizee_fitness_weights))
#             weighted_fitness_list.append(weighted_fitness)
# 
#             # We need to convert the current run index into an ind_idx
#             # (index of individual within one generation)
#             traj.v_idx = run_index
#             ind_index = traj.par.ind_idx
#             individual = old_eval_pop[ind_index]
# 
#             # Accept or reject the new solution
#             current_fitness_value_i = self.current_fitness_value_list[i]
#             r = np.random.rand()
#             p = np.exp((weighted_fitness - current_fitness_value_i) / self.T)
# 
#             # Accept
#             if r < p or weighted_fitness >= current_fitness_value_i:
#                 self.current_fitness_value_list[i] = weighted_fitness
#                 self.current_individual_list[i] = np.array(dict_to_list(individual))
# 
#             traj.f_add_result('$set.$.individual', individual)
#             # Watchout! if weighted fitness is a tuple/np array it should be converted to a list first here
#             traj.f_add_result('$set.$.fitness', weighted_fitness)
# 
#             current_individual = self.current_individual_list[i]
#             new_individual = list_to_dict(current_individual + np.random.randn(current_individual.size) * noisy_step * self.T,
#                                           self.optimizee_individual_dict_spec)
#             if self.optimizee_bounding_func is not None:
#                 new_individual = self.optimizee_bounding_func(new_individual)
# 
#             logger.debug("Current best fitness for individual %d is %.2f. New individual is %s", 
#                          i, self.current_fitness_value_list[i], new_individual)
#             self.eval_pop.append(new_individual)
# 
#         logger.debug("Current best fitness within population is %.2f", max(self.current_fitness_value_list))
# 
#         traj.v_idx = -1  # set the trajectory back to default
#         logger.info("-- End of generation {} --".format(self.g))
# 
#         # ------- Create the next generation by crossover and mutation -------- #
#         # not necessary for the last generation
#         if self.g < n_iteration - 1 and stop_criterion > max(self.current_fitness_value_list):
#             fitnesses_results.clear()
#             self.g += 1  # Update generation counter
#             self._expand_trajectory(traj)

    def end(self):
        """
        See :meth:`~ltl.optimizers.optimizer.Optimizer.end`
        """
        # ------------ Finished all runs and print result --------------- #
        best_last_indiv_index = np.argmax(self.old_fitness_value_list)
        best_last_indiv = self.old_eval_pop[best_last_indiv_index]
        best_last_fitness = self.old_fitness_value_list[best_last_indiv_index]

        logger.info("The best last individual was %s with fitness %s", best_last_indiv, best_last_fitness)
        logger.info("-- End of (successful) annealing --")
        
    def generateIndividuals(self, pop_size, bounds_min, bounds_max):
        
        individuals = np.array(np.random.uniform(bounds_min, bounds_max, (pop_size, self.dimensions)))
        
        new_individual_list = [
            list_to_dict(ind_as_list, self.optimizee_individual_dict_spec)
            for ind_as_list in individuals
        ]
        if self.optimizee_bounding_func is not None:
            new_individual_list = [self.optimizee_bounding_func(ind) for ind in new_individual_list]
            
        return new_individual_list