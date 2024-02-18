import argparse
import glob
import os

import numpy as np
import statsmodels.stats.proportion as smp
from stable_baselines3.common.utils import set_random_seed

from indago.avf.config import (
    AVF_TEST_POLICIES,
    AVF_TRAIN_POLICIES,
    CLASSIFIER_LAYERS,
    CROSSOVER_RATE,
    DNN_SAMPLING_POLICIES,
    POPULATION_SIZE,
)
from indago.config import DONKEY_ENV_NAME, ENV_IDS, ENV_NAMES
from indago.envs.donkey.scenes.simulator_scenes import SIMULATOR_SCENES_DICT
from indago.exp_configurator import ExpConfigurator
from indago.utils.env_utils import ALGOS
from log import Log, close_loggers

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs")
parser.add_argument(
    "--algo",
    help="RL Algorithm",
    default="sac",
    type=str,
    required=False,
    choices=list(ALGOS.keys()),
)
parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
parser.add_argument("--exp-id", help="Experiment ID (0: latest)", default=0, type=int)
parser.add_argument(
    "--env-name", help="Env name", type=str, choices=ENV_NAMES, default="park"
)
parser.add_argument(
    "--env-id", help="Env id", type=str, choices=ENV_IDS, default="parking-v0"
)

parser.add_argument(
    "--num-episodes",
    help="Num episodes (i.e. num of env configurations) to run",
    type=int,
    default=1,
)
parser.add_argument("--failure-prob-dist", action="store_true", default=False)
parser.add_argument(
    "--num-runs-each-env-config",
    help="Num runs for each env configuration (valid when failure probability dist = True)",
    type=int,
    default=30,
)

# AVF params
parser.add_argument(
    "--avf-test-policy",
    help="Avf policy testing",
    type=str,
    choices=AVF_TEST_POLICIES,
    default=None,
)
parser.add_argument(
    "--avf-train-policy",
    help="Avf train policy",
    type=str,
    choices=AVF_TRAIN_POLICIES,
    default="mlp",
)
parser.add_argument("--exp-name", help="Experiment name suffix", type=str, default=None)
parser.add_argument(
    "--dnn-sampling",
    help="Sampling policy when using DNN predictors",
    type=str,
    choices=DNN_SAMPLING_POLICIES,
    default="original",
)
parser.add_argument(
    "--budget",
    help="Timeout in seconds for the failure search technique",
    type=int,
    default=-1,
)
parser.add_argument(
    "--sampling-size",
    help="Sampling size when using original dnn sampling technique",
    type=int,
    default=100000,
)
parser.add_argument(
    "--neighborhood-size",
    help="Neighborhood size when using the hc techniques",
    type=int,
    default=50,
)
parser.add_argument(
    "--hc-counter", help="Stopping counter for the hc techniques", type=int, default=100
)
parser.add_argument(
    "--stagnation-tolerance",
    help="Tolerance when comparing max predictions in the hc techniques",
    type=float,
    default=0.005,
)
parser.add_argument(
    "--population-size",
    help="Population size for ga techniques",
    type=int,
    default=POPULATION_SIZE,
)
parser.add_argument(
    "--crossover-rate",
    help="Mutation rate for ga techniques",
    type=float,
    default=CROSSOVER_RATE,
)
parser.add_argument(
    "--training-progress-filter",
    help="Percentage of training to filter",
    type=int,
    default=10,
)
parser.add_argument(
    "--oversample",
    help="Percentage of oversampling of the minority class for the classification problem",
    type=float,
    default=0.0,
)
parser.add_argument(
    "--layers",
    help="Num layers architecture",
    type=int,
    choices=CLASSIFIER_LAYERS,
    default=1,
)
parser.add_argument("--regression", action="store_true", default=False)
parser.add_argument(
    "--minimize",
    help="Minimize the fitness function (applies when regression is activated)",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--model-checkpoint",
    help="Model checkpoint to load (valid when estimate failure probability = True)",
    type=int,
    default=-1,
)
parser.add_argument(
    "--exp-file",
    help="Experiment file produced with a previous run that shows the failures the agent experienced at test time",
    type=str,
    default=None,
)
parser.add_argument(
    "--parallelize",
    help="Parallelize experiments for failure_probability_estimation (not valid for random and prioritized_replay avf_test_policies)",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--build-denoised-failure-set",
    help="Execute all training failures and build a dataset discarding those in which the trained agent does not fail",
    action="store_true",
    default=False,
)

# DonkeyCar parameters
parser.add_argument(
    "-vae", "--vae-path", help="Path to saved VAE", type=str, default=None
)
parser.add_argument(
    "--add-to-port",
    help="Adding to default port 9091 in order to execute more simulators in parallel",
    type=int,
    default=-1,
)
parser.add_argument(
    "--simulation-mul",
    help="Speed up DonkeyCar simulation by at most 5x",
    type=int,
    default=1,
)
parser.add_argument(
    "--z-size",
    help="Latent space size. Needs to match the latent space of the trained VAE",
    type=int,
    default=64,
)
parser.add_argument(
    "--exe-path", help="DonkeyCar simulator execution path", type=str, default=None
)
parser.add_argument(
    "--remove-road-constraints",
    help="Remove constraints for generating roads (i.e., initial configurations of the driving environment)",
    action="store_true",
    default=False,
)

parser.add_argument(
    "--num-runs-experiments",
    help="Number of times to run the experiments",
    type=int,
    default=1,
)

parser.add_argument(
    "--resume-dir",
    help="Directory where the model retrained with test failures is. "
    "By default the best model from the directory will be loaded.",
    type=str,
    default=None,
)

args = parser.parse_args()


def run_experiments(
    env_name: str,
    env_id: str,
    seed: int,
    regression: bool,
    minimize: bool,
    num_envs: int,
    folder: str,
    algo: str,
    exp_id: int,
    avf_train_policy: str,
    avf_test_policy: str,
    dnn_sampling: str,
    sampling_size: int,
    neighborhood_size: int,
    hc_counter: int,
    stagnation_tolerance: float,
    model_checkpoint: int,
    training_progress_filter: int,
    layers: int,
    oversample_minority_class_percentage: float,
    failure_prob_dist: bool,
    num_episodes: int,
    num_runs_each_env_config: int,
    exp_name: str,
    vae_path: str,
    add_to_port: int,
    simulation_mul: int,
    z_size: int,
    exe_path: str,
    exp_file: str,
    parallelize: bool,
    budget: int,
    population_size: int,
    crossover_rate: float,
    resume_dir: str,
    remove_road_constraints: bool,
) -> None:
    logger = Log(logger_prefix="experiments")
    logger.info("Args: {}".format(args))

    if seed == -1:
        try:
            seed = np.random.randint(2**32 - 1)
        except ValueError:
            seed = np.random.randint(2**30 - 1)

    # also set when instantiating the algorithm
    set_random_seed(seed=seed)

    exp_configurator = ExpConfigurator(
        env_name=env_name,
        env_id=env_id,
        seed=seed,
        regression=regression,
        minimize=minimize,
        num_envs=num_envs,
        folder=folder,
        algo=algo,
        exp_id=exp_id,
        avf_train_policy=avf_train_policy,
        avf_test_policy=avf_test_policy,
        dnn_sampling=dnn_sampling,
        sampling_size=sampling_size,
        neighborhood_size=neighborhood_size,
        stagnation_tolerance=stagnation_tolerance,
        hc_counter=hc_counter,
        model_checkpoint=model_checkpoint,
        training_progress_filter=training_progress_filter,
        layers=layers,
        oversample_minority_class_percentage=oversample_minority_class_percentage,
        failure_prob_dist=failure_prob_dist,
        num_episodes=num_episodes,
        num_runs_each_env_config=num_runs_each_env_config,
        exp_name=exp_name,
        vae_path=vae_path,
        add_to_port=add_to_port,
        simulation_mul=simulation_mul,
        z_size=z_size,
        exe_path=exe_path,
        exp_file=exp_file,
        parallelize_fp_dist=parallelize,
        budget=budget,
        population_size=population_size,
        crossover_rate=crossover_rate,
        resume_dir=resume_dir,
        remove_road_constraints=remove_road_constraints,
    )
    # It will change when exp_file is not None
    num_episodes = exp_configurator.num_episodes

    if failure_prob_dist:
        num_experiments = 0
        num_failures = 0
        previous_env_config = None
        map_env_config_failure_prob = dict()
        map_env_config_min_fitness = dict()
        num_trials = 0
        episode_num = 0
        min_fitness_values = []
        while num_experiments < num_runs_each_env_config * num_episodes:
            if num_experiments % num_runs_each_env_config == 0 and num_experiments != 0:
                map_env_config_failure_prob[previous_env_config.get_str()] = (
                    num_failures / num_runs_each_env_config,
                    smp.proportion_confint(
                        count=num_failures,
                        nobs=num_runs_each_env_config,
                        method="wilson",
                    ),
                )
                logger.info(
                    "Failure probability for env config {}: {}".format(
                        previous_env_config.get_str(),
                        map_env_config_failure_prob[previous_env_config.get_str()],
                    )
                )
                if len(min_fitness_values) > 0:
                    map_env_config_min_fitness[previous_env_config.get_str()] = np.mean(
                        min_fitness_values
                    )
                    min_fitness_values.clear()

                num_failures = 0
                episode_num = 0
                num_trials += 1
            failure, env_config, fitness_values = exp_configurator.test_single_episode(
                episode_num=episode_num, num_trials=num_trials
            )

            logger.debug("Env config: {}".format(env_config.get_str()))

            if len(fitness_values) > 0:
                min_fitness_values.append(min(fitness_values))
                logger.debug(f"Min fitness value: {min(fitness_values)}")

            previous_env_config = env_config
            if failure:
                num_failures += 1
            num_experiments += 1
            episode_num += 1
            logger.debug(
                "Num experiments: {}/{}".format(
                    num_experiments, num_runs_each_env_config * num_episodes
                )
            )

        if len(map_env_config_failure_prob) > 0:
            map_env_config_failure_prob[previous_env_config.get_str()] = (
                num_failures / num_runs_each_env_config,
                smp.proportion_confint(
                    count=num_failures, nobs=num_runs_each_env_config, method="wilson"
                ),
            )

        if len(map_env_config_min_fitness) > 0:
            map_env_config_min_fitness[previous_env_config.get_str()] = np.mean(
                min_fitness_values
            )
            min_fitness_values.clear()

        values = []

        num_failures = 0
        for key, value in map_env_config_failure_prob.items():
            if value[0] > 0.5:
                num_failures += 1
                logger.info(
                    "FAIL - Failure probability for env config {}: {}".format(
                        key, value
                    )
                )
            else:
                logger.info(
                    "Failure probability for env config {}: {}".format(key, value)
                )
            values.append(value[0])

        if len(min_fitness_values) > 0:
            logger.info(
                "Min fitness values: Mean: {:.2f}, Std: {:.2f}, Min: {}, Max: {}, Values: {}".format(
                    np.mean(min_fitness_values),
                    np.std(min_fitness_values),
                    np.min(min_fitness_values),
                    np.max(min_fitness_values),
                    min_fitness_values,
                )
            )
        elif len(map_env_config_min_fitness) > 0:
            mean_fitness_values = [
                mean_fitness_value
                for mean_fitness_value in map_env_config_min_fitness.values()
            ]
            logger.info(
                "Min fitness values: Mean: {:.2f}, Std: {:.2f}, Min: {}, Max: {}, Values: {}".format(
                    np.mean(mean_fitness_values),
                    np.std(mean_fitness_values),
                    np.min(mean_fitness_values),
                    np.max(mean_fitness_values),
                    mean_fitness_values,
                )
            )

        if len(values) > 0:
            logger.info(
                "Failure probabilities: {}, Mean: {:.2f}, Std: {:.2f}, Min: {}, Max: {}".format(
                    values,
                    np.mean(values),
                    np.std(values),
                    np.min(values),
                    np.max(values),
                )
            )

        logger.info("{}".format(map_env_config_failure_prob.keys()))

    logger.info(
        "Number of evaluation predictions (i.e. number of times a model was used to make predictions): {}".format(
            exp_configurator.get_num_evaluation_predictions()
        )
    )
    exp_configurator.close_env()


def sort_filenames(f: str) -> int:
    return int(f.split("-")[-2])


if __name__ == "__main__":

    env_id = args.env_id

    if args.env_name == DONKEY_ENV_NAME:
        simulator_scene = SIMULATOR_SCENES_DICT["generated_track"]
        assert env_id == "DonkeyVAE-v0", "env_id must be DonkeyVAE, found: {}".format(
            env_id
        )
        env_id = "DonkeyVAE-v0-scene-{}".format(simulator_scene.get_scene_name())

    num_envs = 1
    if args.num_runs_experiments == 1:
        if (
            args.exp_file is not None
            and "trial" in args.exp_file
            and args.avf_test_policy == "replay_test_failure"
        ):
            if args.avf_train_policy not in args.exp_file:
                name = args.exp_file.split("-")[1]
                # trying to catch the suffix
                i = 2
                while args.exp_file.split("-")[i] != "failure":
                    name += f"-{args.exp_file.split('-')[i]}"
                    i += 1
                exp_filenames = glob.glob(
                    os.path.join(
                        args.folder,
                        args.algo,
                        env_id + "_" + str(args.exp_id),
                        "testing-{}-*-trial.txt".format(name),
                    )
                )
            else:
                name = args.exp_file.split("-")[2]
                # trying to catch the suffix
                i = 3
                while args.exp_file.split("-")[i] != "failure":
                    name += f"-{args.exp_file.split('-')[i]}"
                    i += 1
                exp_filenames = glob.glob(
                    os.path.join(
                        args.folder,
                        args.algo,
                        env_id + "_" + str(args.exp_id),
                        "testing-{}-{}-*-trial.txt".format(args.avf_train_policy, name),
                    )
                )

            exp_filenames = sorted(exp_filenames, key=sort_filenames)
            assert len(exp_filenames) > 0, "No matches for {}".format(args.exp_file)
            for exp_filename in exp_filenames:
                run_experiments(
                    env_name=args.env_name,
                    env_id=args.env_id,
                    seed=args.seed,
                    regression=args.regression,
                    minimize=args.minimize,
                    num_envs=num_envs,
                    folder=args.folder,
                    algo=args.algo,
                    exp_id=args.exp_id,
                    avf_train_policy=args.avf_train_policy,
                    avf_test_policy=args.avf_test_policy,
                    dnn_sampling=args.dnn_sampling,
                    sampling_size=args.sampling_size,
                    neighborhood_size=args.neighborhood_size,
                    stagnation_tolerance=args.stagnation_tolerance,
                    hc_counter=args.hc_counter,
                    model_checkpoint=args.model_checkpoint,
                    training_progress_filter=args.training_progress_filter,
                    layers=args.layers,
                    oversample_minority_class_percentage=args.oversample,
                    failure_prob_dist=args.failure_prob_dist,
                    num_episodes=args.num_episodes,
                    num_runs_each_env_config=args.num_runs_each_env_config,
                    exp_name=args.exp_name,
                    vae_path=args.vae_path,
                    add_to_port=args.add_to_port,
                    simulation_mul=args.simulation_mul,
                    z_size=args.z_size,
                    exe_path=args.exe_path,
                    exp_file=exp_filename[exp_filename.rindex(os.sep) + 1 :],
                    parallelize=args.parallelize,
                    budget=args.budget,
                    population_size=args.population_size,
                    crossover_rate=args.crossover_rate,
                    resume_dir=args.resume_dir,
                    remove_road_constraints=args.remove_road_constraints,
                )
                close_loggers()
        else:
            run_experiments(
                env_name=args.env_name,
                env_id=args.env_id,
                seed=args.seed,
                regression=args.regression,
                minimize=args.minimize,
                num_envs=num_envs,
                folder=args.folder,
                algo=args.algo,
                exp_id=args.exp_id,
                avf_train_policy=args.avf_train_policy,
                avf_test_policy=args.avf_test_policy,
                dnn_sampling=args.dnn_sampling,
                sampling_size=args.sampling_size,
                neighborhood_size=args.neighborhood_size,
                stagnation_tolerance=args.stagnation_tolerance,
                hc_counter=args.hc_counter,
                model_checkpoint=args.model_checkpoint,
                training_progress_filter=args.training_progress_filter,
                layers=args.layers,
                oversample_minority_class_percentage=args.oversample,
                failure_prob_dist=args.failure_prob_dist,
                num_episodes=args.num_episodes,
                num_runs_each_env_config=args.num_runs_each_env_config,
                exp_name=args.exp_name,
                vae_path=args.vae_path,
                add_to_port=args.add_to_port,
                simulation_mul=args.simulation_mul,
                z_size=args.z_size,
                exe_path=args.exe_path,
                exp_file=args.exp_file,
                parallelize=args.parallelize,
                budget=args.budget,
                population_size=args.population_size,
                crossover_rate=args.crossover_rate,
                resume_dir=args.resume_dir,
                remove_road_constraints=args.remove_road_constraints,
            )
    else:
        for i in range(args.num_runs_experiments):
            run_experiments(
                env_name=args.env_name,
                env_id=args.env_id,
                seed=args.seed,
                regression=args.regression,
                minimize=args.minimize,
                num_envs=num_envs,
                folder=args.folder,
                algo=args.algo,
                exp_id=args.exp_id,
                avf_train_policy=args.avf_train_policy,
                avf_test_policy=args.avf_test_policy,
                dnn_sampling=args.dnn_sampling,
                sampling_size=args.sampling_size,
                neighborhood_size=args.neighborhood_size,
                stagnation_tolerance=args.stagnation_tolerance,
                hc_counter=args.hc_counter,
                model_checkpoint=args.model_checkpoint,
                training_progress_filter=args.training_progress_filter,
                layers=args.layers,
                oversample_minority_class_percentage=args.oversample,
                failure_prob_dist=args.failure_prob_dist,
                num_episodes=args.num_episodes,
                num_runs_each_env_config=args.num_runs_each_env_config,
                exp_name=(
                    "{}-trial".format(i + 1)
                    if args.exp_name is None
                    else "{}-{}-trial".format(args.exp_name, i + 1)
                ),
                vae_path=args.vae_path,
                add_to_port=args.add_to_port,
                simulation_mul=args.simulation_mul,
                z_size=args.z_size,
                exe_path=args.exe_path,
                exp_file=args.exp_file,
                parallelize=args.parallelize,
                budget=args.budget,
                population_size=args.population_size,
                crossover_rate=args.crossover_rate,
                resume_dir=args.resume_dir,
                remove_road_constraints=args.remove_road_constraints,
            )
            close_loggers()
