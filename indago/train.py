# Author: Antonin Raffin

"""
The MIT License

Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import argparse
import glob
import logging
import os
import time
from collections import OrderedDict

import numpy as np
import yaml
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from indago.avf.avf import Avf
from indago.avf.config import AVF_TEST_POLICIES, AVF_TEST_POLICIES_WITH_DNN, AVF_TRAIN_POLICIES
from indago.config import (
    DONKEY_ENV_NAME,
    ENV_IDS,
    ENV_NAMES,
    MAX_CTE_ERROR,
    MAX_DY_VALUE,
    MAX_STEERING_DIFF,
    MAX_THROTTLE,
    MAX_VALUE_COMMANDS,
    MIN_THROTTLE,
    N_COMMAND_HISTORY,
    NUM_COMMANDS_TRACK,
    SIM_PARAMS,
)
from indago.envs.donkey.scenes.simulator_scenes import SIMULATOR_SCENES_DICT
from indago.progress_bar_callback import ProgressBarManager
from indago.save_best_model_callback import SaveBestModelCallback
from indago.save_vec_normalize_callback import SaveVecNormalizeCallback
from indago.utils.dummy_c_vec_env import DummyCVecEnv
from indago.utils.env_utils import ALGOS, constfn, get_latest_run_id, linear_schedule, make_env_fn
from indago.utils.torch_utils import DEVICE
from log import Log

parser = argparse.ArgumentParser()
parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default="", type=str)
parser.add_argument("--algo", help="RL Algorithm", default="sac", type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument("-n", "--n-timesteps", help="Overwrite the number of timesteps", default=-1, type=int)
parser.add_argument("--log-interval", default=-1, type=int)
parser.add_argument("-v", "--verbose", type=int, default=0)
parser.add_argument("--env-name", help="Env name", type=str, choices=ENV_NAMES, default="park")
parser.add_argument("--env-id", help="Env id", type=str, choices=ENV_IDS, default="parking-v0")
parser.add_argument("-f", "--log-folder", help="Log folder", type=str, default="logs")
parser.add_argument("--algo-hyperparams", type=str, default=None)
# DonkeyCar parameters
parser.add_argument("-vae", "--vae-path", help="Path to saved VAE", type=str, default=None)
parser.add_argument(
    "--add-to-port", help="Adding to default port 9091 in order to execute more simulators in parallel", type=int, default=-1
)
parser.add_argument("--simulation-mul", help="Speed up DonkeyCar simulation by at most 5x", type=int, default=1)
parser.add_argument(
    "--z-size", help="Latent space size. Needs to match the latent space of the trained VAE", type=int, default=64
)
parser.add_argument("--exe-path", help="DonkeyCar simulator execution path", type=str, default=None)
parser.add_argument(
    "--headless",
    help="Whether to run the simulator headless or not (only valid for DonkeyCar simulator)",
    action="store_true",
    default=False,
)

parser.add_argument("--n-eval-episodes", type=int, default=1)
parser.add_argument("--no-progress-bar", action="store_true", default=False)
parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
parser.add_argument("--exp-file-folder", help="Log folder where to find the experiment file (exp-file)", type=str, default="logs")
parser.add_argument(
    "--exp-file",
    help="Experiment file produced with a previous run that shows the failures the agent experienced at test time. "
         "Here it is used to resume/restart a training to make sure that the failures at test time are learnable.",
    type=str,
    default=None,
)
parser.add_argument("--exp-id", help="Experiment ID (0: latest) of the agent we want to take the test failures from", default=0, type=int)
parser.add_argument("--avf-train-policy", help="Avf train policy", type=str, choices=AVF_TRAIN_POLICIES, default="mlp")

args, _ = parser.parse_known_args()

logger = Log("train")

if args.seed == -1:
    args.seed = np.random.randint(2 ** 32 - 1)

set_random_seed(args.seed)

assert 1 <= args.simulation_mul <= 5, "Simulation multiplier for DonkeyCar simulator must be in [1, 5], found {}".format(
    args.simulation_mul
)
simulator_scene = SIMULATOR_SCENES_DICT["generated_track"]

logger.info("DEVICE: {}".format(DEVICE))

ENV_ID = args.env_id
if args.env_name == DONKEY_ENV_NAME:
    assert ENV_ID == "DonkeyVAE-v0", "env_id must be DonkeyVAE, found: {}".format(ENV_ID)
    ENV_ID = "DonkeyVAE-v0-scene-{}".format(simulator_scene.get_scene_name())

avf_test_policy = None
exp_filenames = []
if args.exp_file is not None:
    for candidate_avf_test_policy in AVF_TEST_POLICIES:
        if candidate_avf_test_policy == "test" or candidate_avf_test_policy == "replay_test_failure":
            continue

        try:
            idx = args.exp_file.index(candidate_avf_test_policy)
            avf_test_policy = args.exp_file[idx:idx + len(candidate_avf_test_policy)]
            assert candidate_avf_test_policy == avf_test_policy, "The two strings do not match"
            # it means that candidate_avf_test_policy is not a substring of the real avf_test_policy
            if args.exp_file[idx + len(candidate_avf_test_policy)] != "_":
                break
        except ValueError:
            pass

    assert avf_test_policy is not None, "avf_test_policy not assigned"

    if avf_test_policy in AVF_TEST_POLICIES_WITH_DNN:
        exp_filenames = glob.glob(
            os.path.join(
                args.exp_file_folder, args.algo, ENV_ID + '_' + str(args.exp_id),
                "testing-{}-{}-*-trial.txt".format(args.avf_train_policy, avf_test_policy)
            )
        )
        if len(exp_filenames) == 0:
            exp_filenames = glob.glob(
                os.path.join(
                    args.exp_file_folder,
                    "testing-{}-{}-*-trial.txt".format(args.avf_train_policy, avf_test_policy)
                )
            )
    else:
        exp_filenames = glob.glob(
            os.path.join(
                args.exp_file_folder, args.algo, ENV_ID + '_' + str(args.exp_id),
                "testing-{}-*-trial.txt".format(avf_test_policy)
            )
        )
        if len(exp_filenames) == 0:
            exp_filenames = glob.glob(
                os.path.join(
                    args.exp_file_folder,
                    "testing-{}-*-trial.txt".format(avf_test_policy)
                )
            )

    assert len(exp_filenames) > 0, "No match found"

tensorboard_log = None if args.tensorboard_log == "" else os.path.join(args.tensorboard_log, ENV_ID)
if args.exp_file is not None:
    tensorboard_log = os.path.join(args.tensorboard_log, "resume_{}_{}_{}".format(avf_test_policy, ENV_ID, args.exp_id))
    logger.info('Retraining agent using the failures produced by {}'.format(avf_test_policy))

# Compute and create log path
log_path = os.path.join(args.log_folder, args.algo)
save_path = os.path.join(log_path, "{}_{}".format(ENV_ID, get_latest_run_id(log_path, ENV_ID) + 1))
if args.exp_file is not None:
    save_path = os.path.join(log_path, "resume_{}_{}_{}_seed_{}".format(avf_test_policy, ENV_ID, args.exp_id, args.seed))

params_path = os.path.join(save_path, ENV_ID)
os.makedirs(params_path, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(
        save_path, "train.txt" if args.exp_file is None else "train_restart_{}.txt".format(avf_test_policy)
    ),
    filemode="w",
    level=logging.DEBUG,
)

logger.info("{} {} {} {}".format("=" * 10, ENV_ID, args.algo, "=" * 10))

if args.algo_hyperparams:
    logger.info("Command line hyperparams: {}".format(args.algo_hyperparams))
    hyperparams = eval(args.algo_hyperparams)
else:
    # Load hyperparameters from yaml file
    with open(os.path.join("hyperparams", "indago", "{}.yml".format(args.algo)), "r") as f:
        hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)[args.env_id]

# Sort hyperparams that will be saved
saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
saved_hyperparams["seed"] = args.seed

if args.env_name == DONKEY_ENV_NAME:
    saved_hyperparams["vae_path"] = args.vae_path
    saved_hyperparams["vae_latent_space_size"] = args.z_size
    for key in SIM_PARAMS:
        saved_hyperparams[key] = eval(key)

# Create learning rate schedules for ppo and sac
if args.algo in ["ppo", "sac", "tqc"]:
    for key in ["learning_rate", "clip_range"]:
        if key not in hyperparams:
            continue
        if isinstance(hyperparams[key], str):
            schedule, initial_value = hyperparams[key].split("_")
            initial_value = float(initial_value)
            hyperparams[key] = linear_schedule(initial_value)
        elif isinstance(hyperparams[key], float):
            hyperparams[key] = constfn(hyperparams[key])
        else:
            raise ValueError("Invalid valid for {}: {}".format(key, hyperparams[key]))

n_envs = hyperparams.pop("n_envs", 1)
assert n_envs == 1, "Num envs {} not supported".format(n_envs)
if args.n_timesteps > 0:
    n_timesteps = args.n_timesteps
    logger.info("Overriding n_timesteps in hyperparams file")
else:
    n_timesteps = int(hyperparams["n_timesteps"])
del hyperparams["n_timesteps"]


if args.log_interval == -1:
    args.log_interval = min(int(n_timesteps * 1 / 100), 1000)
    logger.info("Logging every {} timesteps".format(args.log_interval))

save_checkpoint_interval = 20
logger.info("Saving checkpoint every {} timesteps".format(save_checkpoint_interval * args.log_interval))

normalize = False
normalize_kwargs = {}
if "normalize" in hyperparams.keys():
    normalize = hyperparams["normalize"]
    if isinstance(normalize, str):
        normalize_kwargs = eval(normalize)
        normalize = True
    del hyperparams["normalize"]

if "policy_kwargs" in hyperparams.keys():
    hyperparams["policy_kwargs"] = eval(hyperparams["policy_kwargs"])

eval_env = None

avf = Avf(env_name=args.env_name, log_path=save_path, storing_logs=True, seed=args.seed, exp_files=exp_filenames)

n_stack = 1
if hyperparams.get("frame_stack", False):
    n_stack = hyperparams["frame_stack"]
    del hyperparams["frame_stack"]

env = DummyCVecEnv(
    env_fns=[
        make_env_fn(
            env_name=args.env_name,
            seed=args.seed,
            save_path=save_path,
            avf=avf,
            vae_path=args.vae_path,
            add_to_port=args.add_to_port,
            simulation_mul=args.simulation_mul,
            z_size=args.z_size,
            n_stack=n_stack,
            exe_path=args.exe_path,
            simulator_scene=simulator_scene,
            headless=args.headless,
        )
    ]
)

if normalize:
    if hyperparams.get("normalize", False) and args.algo in ["ddpg"]:
        logger.info("WARNING: normalization not supported yet for DDPG")
    else:
        logger.info("Normalizing input and return")
        env = VecNormalize(env, **normalize_kwargs)

# Parse noise string for DDPG and SAC
if args.algo in ["ddpg", "sac", "tqc"] and hyperparams.get("noise_type") is not None:
    noise_type = hyperparams["noise_type"].strip()
    noise_std = hyperparams["noise_std"]
    n_actions = env.action_space.shape[0]
    if "normal" in noise_type:
        hyperparams["action_noise"] = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
    elif "ornstein-uhlenbeck" in noise_type:
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
        )
    else:
        raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
    logger.info("Applying {} noise with std {}".format(noise_type, noise_std))
    del hyperparams["noise_type"]
    del hyperparams["noise_std"]

# Pre-process train_freq
if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
    hyperparams["train_freq"] = tuple(hyperparams["train_freq"])

if "model_class" in hyperparams:
    if hyperparams["model_class"] == "tqc" or hyperparams["model_class"] == "sac":
        hyperparams["model_class"] = ALGOS[hyperparams["model_class"]]
    else:
        raise NotImplementedError("Model class {} for HER not supported".format(hyperparams["model_class"]))

if args.exp_file is not None:
    model_path = os.path.join(args.exp_file_folder, args.algo, ENV_ID + '_' + str(args.exp_id), 'best_model.zip')
    if not os.path.exists(model_path):
        model_path = os.path.join(args.exp_file_folder, 'best_model.zip')
    logger.info("Resuming training from model {}".format(model_path))
    model = ALGOS[args.algo].load(path=model_path, tensorboard_log=tensorboard_log, env=env, seed=args.seed)
else:
    model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log, seed=args.seed, verbose=args.verbose, **hyperparams)

verbose = 1

save_best_model_callback = SaveBestModelCallback(
    log_interval=args.log_interval,
    save_checkpoint_interval=save_checkpoint_interval,
    log_dir=save_path,
    verbose=verbose,
    num_envs=n_envs,
    eval_env=eval_env,
    normalize_kwargs=normalize_kwargs,
    n_eval_episodes=args.n_eval_episodes,
    avf=avf if args.exp_file is not None else None
)

start_learning_time = time.time()

# Save hyperparams
with open(os.path.join(params_path, "hyperparams.yml"), "w") as f:
    yaml.dump(saved_hyperparams, f)

logger.info("******* Start learning *******")

callbacks = []
if normalize:
    callbacks.append(SaveVecNormalizeCallback(log_interval=args.log_interval, num_envs=n_envs, save_path=save_path))

callbacks.append(save_best_model_callback)
try:
    if not args.no_progress_bar:
        with ProgressBarManager(total_timesteps=n_timesteps) as progress_callback:
            callbacks.append(progress_callback)
            model.learn(n_timesteps, callback=callbacks)
    else:
        model.learn(n_timesteps, callback=callbacks)
finally:
    if len(normalize_kwargs) > 0:
        # Important: save the running average, for testing the agent we need that normalization
        model.get_vec_normalize_env().save(os.path.join(save_path, "vecnormalize.pkl"))

logger.info("******* End learning *******")

logger.info("Time taken: {} s".format((time.time() - start_learning_time)))

if args.exp_file is not None:
    non_learnable_failure_configurations = [
        avf.failure_test_env_configs[index] for index in save_best_model_callback.indices_failing_configurations
    ]
    logger.info("# Non learnable failure configurations: {}".format(len(non_learnable_failure_configurations)))
    for non_learnable_failure_configuration in non_learnable_failure_configurations:
        logger.info(non_learnable_failure_configuration.get_str())

# Close the connection properly
env.reset()
if isinstance(env, VecFrameStack):
    env = env.venv

# HACK to bypass Monitor wrapper
if args.env_name == DONKEY_ENV_NAME:
    env.envs[0].env.exit_scene()
    time.sleep(5)
    env.envs[0].env.close_connection()
    if args.exe_path:
        env.envs[0].close()
else:
    env.close()
