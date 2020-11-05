from datetime import datetime
from getpass import getuser

from l2l.optimizees.comet.optimizee import CometOptimizee, \
    CometOptimizeeParameters
from l2l.optimizers.evolution import GeneticAlgorithmOptimizer,\
    GeneticAlgorithmParameters
from l2l.utils.experiment import Experiment

from comet.evaluation.joint_test import joint_test

import argparse


def run_experiment(args):
    experiment = Experiment(root_dir_path='../results')
    name = 'L2L-COMET-{}-{}'.format(getuser(),
                                    datetime.now().strftime("%Y-%m-%d-%H_%M"))
    trajectory_name = 'comet'
    jube_params = {"exec": "srun -N 1 -n 1 -c 8 python"}
    traj, params = experiment.prepare_experiment(trajectory_name=trajectory_name,
                                                 jube_parameter=jube_params, name=name)

    # Resolve model and noise sources from input arguments
    if args.model == 'brunel':
        if args.noise_type == 'poisson':
            from comet_brunel_hyperparams import optimizee_params
            from comet_brunel_hyperparams import optimizer_params
            from comet.models.brunel.model_params import net_dict, bounds_dict
            from comet.models.brunel.brunel_model import brunel_model as sim_model
        elif args.noise_type == 'pink':
            raise NotImplementedError('No pink noise for brunel model')
        elif args.model == 'microcircuit':
            if args.noise_type == 'poisson':
                from comet_microcircuit_hyperparams import optimizee_params
                from comet_microcircuit_hyperparams import optimizer_params
                from comet.models.microcircuit.model_params import net_dict, bounds_dict
                from comet.models.microcircuit.microcircuit_model import microcircuit_model as sim_model
            elif args.noise_type == 'pink':
                raise NotImplementedError('To be implemented')

    # Resolve mode arg
    if args.mode == 'syn':
        exp_model = None
    elif args.mode == 'exp':
        raise NotImplementedError('To be implemented')

    # Set up optimizee
    optimizee_parameters = CometOptimizeeParameters(
        seed=optimizee_params['seed'],
        keys_to_evolve=optimizee_params['keys_to_evolve'],
        default_params_dict=net_dict,
        default_bounds_dict=bounds_dict,
        model_class=sim_model,
        experiment_class=exp_model,
        test_class=joint_test)
    # Inner-loop simulator
    optimizee = CometOptimizee(traj, optimizee_parameters)

    # Outer-loop optimizer initialization
    optimizer_parameters = GeneticAlgorithmParameters(
        popsize=optimizer_params['popsize'],
        seed=optimizer_params['seed'],
        CXPB=optimizer_params['CXPB'],
        MUTPB=optimizer_params['MUTPB'],
        NGEN=optimizer_params['NGEN'],
        indpb=optimizer_params['indpb'],
        tournsize=optimizer_params['tournsize'],
        matepar=optimizer_params['matepar'],
        mutpar=optimizer_params['mutpar'])

    optimizer = GeneticAlgorithmOptimizer(traj,
                                          optimizee_create_individual=optimizee.create_individual,
                                          parameters=optimizer_parameters,
                                          optimizee_bounding_func=optimizee.bounding_func,
                                          optimizee_fitness_weights=(-1,))

    # Add post processing
    experiment.run_experiment(optimizee=optimizee,
                              optimizee_parameters=optimizee_parameters,
                              optimizer=optimizer,
                              optimizer_parameters=optimizer_parameters)
    traj, paths = experiment.end(optimizer)
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
