import logging
import os
from typing import Tuple

import yaml
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from indago.avf.avf import Avf
from indago.config import DONKEY_ENV_NAME, PARK_ENV_NAME
from indago.envs.donkey.scenes.simulator_scenes import SIMULATOR_SCENES_DICT
from indago.utils.dummy_c_vec_env import DummyCVecEnv
from indago.utils.env_utils import ALGOS, get_latest_run_id, make_env_fn
from indago.utils.file_utils import get_training_logs_path
from log import Log


def instantiated_avf(
        env_name: str,
        algo: str,
        folder: str,
        seed: int,
        env_id: str,
        exp_id: int,
        regression: bool = False,
        exp_file: str = None,
        resume_dir: str = None
) -> Tuple[str, Avf]:

    log_path = get_training_logs_path(folder=folder, algo=algo, env_id=env_id, exp_id=exp_id, resume_dir=resume_dir)
    avf = Avf(env_name=env_name, log_path=log_path, storing_logs=False, seed=seed, regression=regression, exp_file=exp_file)

    return log_path, avf


def instantiate_trained_policy(
    env_name: str,
    algo: str,
    folder: str,
    seed: int,
    env_id: str,
    exp_id: int,
    model_checkpoint: int = -1,
    headless: bool = False,
    enjoy_mode: bool = True,
    testing_strategy: str = None,
    regression: bool = False,
    vae_path: str = None,
    add_to_port: int = 0,
    simulation_mul: int = 1,
    z_size: int = 64,
    exe_path: str = None,
    exp_file: str = None,
    resume_dir: str = None
) -> Tuple[BaseAlgorithm, Avf, VecEnv, str, str]:

    logger = Log("instantiate_model")

    assert 1 <= simulation_mul <= 5, "Simulation multiplier for DonkeyCar simulator must be in [1, 5], found {}".format(
        simulation_mul
    )
    simulator_scene = SIMULATOR_SCENES_DICT["generated_track"]

    # Load hyperparameters from yaml file
    with open(os.path.join("hyperparams", "indago", "{}.yml".format(algo)), "r") as f:
        hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)[env_id]

    n_stack = 1
    if hyperparams.get("frame_stack", False):
        n_stack = hyperparams["frame_stack"]

    if env_name == DONKEY_ENV_NAME:
        assert env_id == "DonkeyVAE-v0", "env_id must be DonkeyVAE, found: {}".format(env_id)
        env_id = "DonkeyVAE-v0-scene-{}".format(simulator_scene.get_scene_name())

    normalize = False
    normalize_kwargs = {}
    if "normalize" in hyperparams.keys():
        normalize = hyperparams["normalize"]
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True
        del hyperparams["normalize"]

    assert exp_id >= 0, "exp_id should be >= 0: {}".format(exp_id)
    if exp_id == 0:
        exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        logger.info("Loading latest experiment, id={}".format(exp_id))

    log_path, avf = instantiated_avf(
        env_name=env_name,
        algo=algo,
        folder=folder,
        seed=seed,
        env_id=env_id,
        exp_id=exp_id,
        regression=regression,
        exp_file=exp_file,
        resume_dir=resume_dir
    )

    if enjoy_mode:
        logging.basicConfig(
            filename=os.path.join(
                log_path, "enjoy_best.txt" if model_checkpoint == -1 else "enjoy_checkpoint_{}.txt".format(model_checkpoint)
            ),
            filemode="w",
            level=logging.DEBUG,
        )
    else:
        assert testing_strategy is not None, "Testing strategy not assigned"

        if exp_file is not None:
            testing_strategy += (
                "_" + exp_file[exp_file.rindex(os.sep) : exp_file.rindex(".")].replace("testing-", "")
                if os.sep in exp_file
                else "_" + exp_file[: exp_file.rindex(".")].replace("testing-", "")
            )

        logging.basicConfig(
            filename=os.path.join(log_path, "testing-{}.txt".format(testing_strategy)), filemode="w", level=logging.DEBUG
        )

    if model_checkpoint != -1:
        model_path = os.path.join(log_path, "model-checkpoint-{}.zip".format(model_checkpoint))
    else:
        model_path = os.path.join(log_path, "best_model.zip")

    logger.info("Loading model: {}".format(model_path))
    logger.info(
        "Setting random seed: {}".format(seed)
    )  # it is here so that it appears in the logs (seed is set in experiments.py)

    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)
    assert os.path.isfile(model_path), "No model found for {} on {}, path: {}".format(algo, env_id, model_path)

    if env_name == DONKEY_ENV_NAME:
        # the first time the environment is instantiated it needs a track which can be any track, since the first
        # episode will override it
        avf.avf_test_policy = "random"

    if not headless and env_name == PARK_ENV_NAME:
        counter = 0
        while os.path.exists(os.path.join(log_path, "videos_{}".format(counter))):
            counter += 1
            os.path.exists(os.path.join(log_path, "videos_{}".format(counter)))
        os.makedirs(name=os.path.join(log_path, "videos_{}".format(counter)))
        log_path = os.path.join(log_path, "videos_{}".format(counter))

    env = DummyCVecEnv(
        env_fns=[
            make_env_fn(
                env_name=env_name,
                seed=seed,
                record_video=not headless,
                avf=avf,
                vae_path=vae_path,
                add_to_port=add_to_port,
                simulation_mul=simulation_mul,
                z_size=z_size,
                n_stack=n_stack,
                exe_path=exe_path,
                simulator_scene=simulator_scene,
                headless=headless,
            )
        ]
    )

    if normalize:
        if hyperparams.get("normalize", False) and algo in ["ddpg"]:
            logger.warn("WARNING: normalization not supported yet for DDPG")
        else:
            logger.debug("Normalizing input and return")
            env = VecNormalize(env, training=False, **normalize_kwargs)
            env = VecNormalize.load(os.path.join(log_path, "vecnormalize.pkl"), env)
            # Deactivate training and reward normalization
            env.training = False
            env.norm_reward = False

    # sets the seed
    return ALGOS[algo].load(path=model_path, env=env, seed=seed), avf, env, log_path, testing_strategy
