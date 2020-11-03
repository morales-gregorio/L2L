import logging.config
import os

from datetime import datetime
from getpass import getuser

from l2l.utils.environment import Environment

from l2l.logging_tools import create_shared_logger_data, configure_loggers
# from l2l.optimizees.comet.optimizee import CometSyntheticOptimizee, \
#     CometSyntheticOptimizeeParameters
from l2l.optimizees.comet.optimizee_synthetic import \
    CometSyntheticOptimizee as Optimizee
from l2l.optimizees.comet.optimizee_experimental import \
    CometsyntheticOptimizeeParameters as OptimizeeParameters
from l2l.optimizers.evolution import GeneticAlgorithmOptimizer,\
    GeneticAlgorithmParameters
from l2l.paths import Paths
import l2l.utils.JUBE_runner as jube

from run_params import optimizee_params, optimizer_params
from comet.models.brunel.model_params import net_dict, bounds_dict
from comet.models.brunel.brunel_model import brunel_model
from comet.evaluation.joint_test import joint_test

logger = logging.getLogger('bin.l2l-comet')


def run_experiment():
    name = 'L2L-COMET-{}-{}'.format(getuser(),
                                    datetime.now().strftime("%Y-%m-%d-%H_%M"))
    try:
        with open('path.conf') as f:
            root_dir_path = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            "You have not set the root path to store your results."
            " Write the path to a path.conf text file in the bin directory"
            " before running the simulation")

    trajectory_name = 'comet'

    paths = Paths(name, dict(run_num='test'), root_dir_path=root_dir_path,
                  suffix="-" + trajectory_name)

    print("All output logs can be found in directory ", paths.logs_path)

    # Create an environment that handles running our simulation
    # This initializes an environment
    env = Environment(
        trajectory=trajectory_name,
        filename=paths.output_dir_path,
        file_title='{} data'.format(name),
        comment='{} data'.format(name),
        add_time=True,
        automatic_storing=True,
        log_stdout=False,  # Sends stdout to logs
    )

    create_shared_logger_data(
        logger_names=['bin', 'optimizers'],
        log_levels=['INFO', 'INFO'],
        log_to_consoles=[True, True],
        sim_name=name,
        log_directory=paths.logs_path)
    configure_loggers()

    # Get the trajectory from the environment
    traj = env.trajectory

    # Set JUBE params
    traj.f_add_parameter_group("JUBE_params", "Contains JUBE parameters")

    # Scheduler parameters
    # Name of the scheduler
    # traj.f_add_parameter_to_group("JUBE_params", "scheduler", "Slurm")
    # Command to submit jobs to the schedulers
    traj.f_add_parameter_to_group("JUBE_params", "submit_cmd", "sbatch")
    # Template file for the particular scheduler
    traj.f_add_parameter_to_group("JUBE_params", "job_file", "job.run")
    # Number of nodes to request for each run
    traj.f_add_parameter_to_group("JUBE_params", "nodes", "1")
    # Requested time for the compute resources
    traj.f_add_parameter_to_group("JUBE_params", "walltime", "01:00:00")
    # MPI Processes per node
    traj.f_add_parameter_to_group("JUBE_params", "ppn", "1")
    # CPU cores per MPI process
    traj.f_add_parameter_to_group("JUBE_params", "cpu_pp", "1")
    # Threads per process
    traj.f_add_parameter_to_group("JUBE_params", "threads_pp", "4")
    # Type of emails to be sent from the scheduler
    traj.f_add_parameter_to_group("JUBE_params", "mail_mode", "ALL")
    # Email to notify events from the scheduler
    traj.f_add_parameter_to_group("JUBE_params", "mail_address",
                                  "x@fz-juelich.de")
    # Error file for the job
    traj.f_add_parameter_to_group("JUBE_params", "err_file", "stderr")
    # Output file for the job
    traj.f_add_parameter_to_group("JUBE_params", "out_file", "stdout")
    # JUBE parameters for multiprocessing. Relevant even without scheduler.
    # MPI Processes per job
    traj.f_add_parameter_to_group("JUBE_params", "tasks_per_job", "1")
    # The execution command
    traj.f_add_parameter_to_group("JUBE_params", "exec",
                                  "srun -n 1 -c 8 --exclusive python " +
                                  os.path.join(paths.root_dir_path,
                                               "run_files/run_optimizee.py"))
    # Ready file for a generation
    traj.f_add_parameter_to_group("JUBE_params", "ready_file",
                                  os.path.join(paths.root_dir_path,
                                               "ready_files/ready_w_"))
    # Path where the job will be executed
    traj.f_add_parameter_to_group("JUBE_params", "work_path",
                                  paths.root_dir_path)
    traj.f_add_parameter_to_group("JUBE_params", "paths_obj", paths)

    # Optimizee params
    # optimizee_seed = 123
    # Keys to evolve
    # TODO: Find out if one key can refer to an entire array
    # keys = ['P_EE', 'P_EI', 'P_IE', 'P_II']
    # optimizee_parameters = CometSyntheticOptimizeeParameters(
    #     seed=optimizee_seed,
    #     keys_to_evolve=keys,
    #     default_params_dict=net_dict,
    #     default_bounds_dict=bounds_dict,
    #     model_class=brunel_model,
    #     test_class=joint_test)
    optimizee_parameters = OptimizeeParameters(
        seed=optimizee_params['seed'],
        keys_to_evolve=optimizee_params['keys_to_evolve'],
        default_params_dict=net_dict,
        default_bounds_dict=bounds_dict,
        model_class=brunel_model,
        test_class=joint_test)
    # Inner-loop simulator
    optimizee = Optimizee(traj, optimizee_parameters)
    jube.prepare_optimizee(optimizee, paths.root_dir_path)

    logger.info("Optimizee parameters: %s", optimizee_parameters)

    # Outer-loop optimizer initialization
    # optimizer_seed = 123
    # pop_size = 24
    # optimizer_parameters = GeneticAlgorithmParameters(popsize=pop_size,
    #                                                   seed=optimizer_seed,
    #                                                   CXPB=0.8,
    #                                                   MUTPB=0.002,
    #                                                   NGEN=50,
    #                                                   indpb=0.2,
    #                                                   tournsize=3,
    #                                                   matepar=0.5,
    #                                                   mutpar=0.05
    #                                                   )
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
    logger.info("Optimizer parameters: %s", optimizer_parameters)

    optimizer = GeneticAlgorithmOptimizer(traj,
                                          optimizee_create_individual=optimizee.create_individual,
                                          parameters=optimizer_parameters,
                                          optimizee_bounding_func=optimizee.bounding_func,
                                          optimizee_fitness_weights=(-1,))

    # Add post processing
    env.add_postprocessing(optimizer.post_process)

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)

    # Outer-loop optimizer end
    optimizer.end()

    # Finally disable logging and close all log-files
    env.disable_logging()

    return traj.v_storage_service.filename, traj.v_name, paths


def main():
    filename, trajname, paths = run_experiment()
    logger.info("Plotting now")


if __name__ == '__main__':
    main()
