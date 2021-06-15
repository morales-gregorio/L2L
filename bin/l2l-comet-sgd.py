from datetime import datetime
from getpass import getuser
from l2l.optimizees.comet.optimizee import CometOptimizee, \
    CometOptimizeeParameters
from l2l.optimizers.gradientdescent.optimizer import GradientDescentOptimizer
from l2l.optimizers.gradientdescent.optimizer import StochasticGDParameters
from l2l.utils.experiment import Experiment
from comet.evaluation.joint_test import joint_test as test_class
import os
from os.path import join
import pandas as pd
import numpy as np
import argparse


def run_experiment(args):

    # Resolve model and noise sources from input arguments
    if args.model == 'brunel':
        if args.noise_type == 'poisson':
            from comet_brunel_hyperparams import optimizee_params, \
                optimizer_params
            from comet.models.brunel.model_params import net_dict, bounds_dict
            from comet.models.brunel.brunel_model import brunel_model as sim_model
        elif args.noise_type == 'pink':
            raise NotImplementedError('No pink noise for brunel model')
    elif args.model == 'microcircuit':
        if args.noise_type == 'poisson':
            from comet_microcircuit_hyperparams import optimizee_params, \
                optimizer_params
            from comet.models.microcircuit.model_params import net_dict, bounds_dict
            from comet.models.microcircuit.microcircuit_model import microcircuit_model as sim_model
        elif args.noise_type == 'pink':
            raise NotImplementedError('To be implemented')

    # Create experiment class, that deals with jobs and submission
    results_dir = '/users/morales/comet2ltl/results'  # XXX Hard coded
    experiment = Experiment(root_dir_path=results_dir)
    name = 'L2L-COMET-SGD-{}-{}'.format(getuser(),
                                    datetime.now().strftime("%Y-%m-%d-%H_%M"))
    os.mkdir(join(results_dir, name))  # Pre-create the results directory
    trajectory_name = 'comet'
    jube_params = \
        {"exec": f"srun -n 1 -c {optimizee_params['threads']} --exclusive python"}
    traj, params = experiment.prepare_experiment(
        trajectory_name=trajectory_name,
        jube_parameter=jube_params, name=name)

    # Create target predictions (either synthetic data or experimental)
    predictions_csv = join(results_dir, name, 'target_predictions.csv')
    if args.mode == 'syn':
        # Calculate target predictions
        target = sim_model(name='Synthetic target',
                           run_params={'seed': optimizee_params['seed'],
                                       'total_num_virtual_procs':
                                           optimizee_params['threads']})
        print('Calculating the default model predictions.')
        test = test_class()
        target_prediction = test.generate_prediction(target)

        # Save predictions to results directory
        df = pd.DataFrame(data=target_prediction.T,
                          columns=[t.name for t in test.test_list],
                          index=np.arange(target_prediction.shape[1]))
        df.to_csv(predictions_csv, index=False)

    elif args.mode == 'exp':
        raise NotImplementedError('To be implemented')

    # Set up optimizee
    optimizee_parameters = CometOptimizeeParameters(
        seed=optimizee_params['seed'],
        threads=optimizee_params['threads'],
        keys_to_evolve=optimizee_params['keys_to_evolve'],
        default_params_dict=net_dict,
        default_bounds_dict=bounds_dict,
        model_class=sim_model,
        target_class=sim_model,
        target_predictions_csv=predictions_csv,
        test_class=test_class)

    # Inner-loop simulator
    optimizee = CometOptimizee(traj, optimizee_parameters)

    # Outer-loop optimizer initialization
    optimizer_parameters = StochasticGDParameters(
        learning_rate=0.01,
        stochastic_deviation=1,
        stochastic_decay=0.99,
        exploration_step_size=0.01,
        n_random_steps=64,  # ~ Population size
        n_iteration=100,
        stop_criterion=-np.inf,
        seed=1234)

    optimizer = GradientDescentOptimizer(
        traj,
        optimizee_create_individual=optimizee.create_individual,
        optimizee_fitness_weights=(-1.,),
        parameters=optimizer_parameters,
        optimizee_bounding_func=optimizee.bounding_func)

    # Add post processing
    experiment.run_experiment(optimizee=optimizee,
                              optimizee_parameters=optimizee_parameters,
                              optimizer=optimizer,
                              optimizer_parameters=optimizer_parameters)
    traj, paths = experiment.end_experiment(optimizer)
    return traj.v_storage_service.filename, traj.v_name, paths


def main(args):
    filename, trajname, paths = run_experiment(args)


if __name__ == '__main__':
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument("--model", choices=['brunel', 'microcircuit'],
                        required=True,
                        help="Network model of choice")
    parser.add_argument("--noise_type", choices=['poisson', 'pink'],
                        required=True,
                        help="External noise input type")
    parser.add_argument("--mode", choices=['syn', 'exp'], required=True,
                        help="""[syn] run optimization on synthetic data,
                                [exp] run optimization on experimental data""")
    parser.add_argument("--area", nargs='?', type=str, required=False,
                        help="[str] area name")
    args = parser.parse_args()
    if args.mode == 'exp' and args.area is None:
        parser.error("--area needs to be specified if --mode is 'exp'")
    main(args)
