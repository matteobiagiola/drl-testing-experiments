import os

import numpy as np
from PIL import Image

from indago.avf.dataset import Data, Dataset
from indago.avf.testing_logs import TestingLogs
from indago.avf.training_logs import TrainingLogs
from indago.config import DONKEY_ENV_NAME, ENV_NAMES, HUMANOID_ENV_NAME, PARK_ENV_NAME
from indago.envs.donkey.donkey_dataset import DonkeyDataset
from indago.envs.donkey.donkey_env_configuration import DonkeyEnvConfiguration
from indago.envs.donkey.donkey_training_logs import DonkeyTrainingLogs
from indago.envs.humanoid.humanoid_dataset import HumanoidDataset
from indago.envs.humanoid.humanoid_env_configuration import HumanoidEnvConfiguration
from indago.envs.humanoid.humanoid_training_logs import HumanoidTrainingLogs
from indago.envs.park.parking_dataset import ParkingDataset
from indago.envs.park.parking_env_configuration import ParkingEnvConfiguration
from indago.envs.park.parking_training_logs import ParkingTrainingLogs
from indago.utils.file_utils import read_logs
from log import Log

logger = Log("preprocessor")


def parse_training_logs(env_name: str, json_data) -> TrainingLogs:
    if env_name == PARK_ENV_NAME or env_name == HUMANOID_ENV_NAME or env_name == DONKEY_ENV_NAME:
        env_config = json_data["env_config"]
        del json_data["env_config"]
        if json_data["agent_state"] is None:
            json_data["agent_state"] = dict()
        if env_name == PARK_ENV_NAME:
            return ParkingTrainingLogs(config=ParkingEnvConfiguration(**env_config), **json_data)
        elif env_name == HUMANOID_ENV_NAME:
            return HumanoidTrainingLogs(config=HumanoidEnvConfiguration(**env_config), **json_data)
        elif env_name == DONKEY_ENV_NAME:
            donkey_env_configuration = DonkeyEnvConfiguration()
            return DonkeyTrainingLogs(config=donkey_env_configuration.str_to_config(s=env_config), **json_data)
    else:
        raise NotImplementedError("Unknown env name: {}".format(env_name))


def parse_testing_logs(env_name: str, json_data) -> TestingLogs:
    if env_name == PARK_ENV_NAME or env_name == HUMANOID_ENV_NAME or env_name == DONKEY_ENV_NAME:
        env_config = json_data["env_config"]
        dynamic_info = json_data["dynamic_info"]
        # TODO: refactor
        if env_name == PARK_ENV_NAME:
            parking_env_configuration = ParkingEnvConfiguration()
            return TestingLogs(config=parking_env_configuration.str_to_config(s=env_config), dynamic_info=dynamic_info)
        elif env_name == HUMANOID_ENV_NAME:
            humanoid_env_configuration = HumanoidEnvConfiguration()
            return TestingLogs(config=humanoid_env_configuration.str_to_config(s=env_config), dynamic_info=dynamic_info)
        elif env_name == DONKEY_ENV_NAME:
            donkey_env_configuration = DonkeyEnvConfiguration()
            return TestingLogs(config=donkey_env_configuration.str_to_config(s=env_config), dynamic_info=dynamic_info)
    else:
        raise NotImplementedError("Unknown env name: {}".format(env_name))


def sort_training_progress(env_name: str, log_path: str, filename: str) -> float:
    if filename.endswith(".json"):
        json_data = read_logs(log_path=log_path, filename=filename)
        training_logs = parse_training_logs(env_name=env_name, json_data=json_data)
        return training_logs.get_training_progress()
    return 0.0


def preprocess_data(
    env_name: str, log_path: str, training_progress_filter: int, policy: str, save_data: bool = False,
) -> Dataset:
    assert os.path.exists(log_path), "Log path {} does not exist".format(log_path)
    assert env_name in ENV_NAMES, "Unknown env name: {}".format(env_name)

    if save_data:
        preprocessed_files_filepath = os.path.join(log_path, "preprocessed")
        if os.path.exists(preprocessed_files_filepath):
            for filename in os.listdir(preprocessed_files_filepath):
                os.remove(os.path.join(preprocessed_files_filepath, filename))

        os.makedirs(preprocessed_files_filepath, exist_ok=True)
        os.makedirs(os.path.join(log_path, "preprocessed", "succeeded"), exist_ok=True)
        os.makedirs(os.path.join(log_path, "preprocessed", "failed"), exist_ok=True)

    if env_name == PARK_ENV_NAME:
        dataset = ParkingDataset(policy=policy)
    elif env_name == HUMANOID_ENV_NAME:
        dataset = HumanoidDataset(policy=policy)
    elif env_name == DONKEY_ENV_NAME:
        dataset = DonkeyDataset(policy=policy)
    else:
        raise NotImplementedError("Unknown env name: {}".format(env_name))

    # TODO: refactor and do this in each training RL algo
    # to estimate the max ent_coef (assuming this is the name of the exploration param in all algos)
    max_exploration_coef = 0.0
    count = 0
    num_failures = 0
    file_suffices = []

    sorted_by_training_progress = sorted(
        os.listdir(path=log_path), key=lambda f: sort_training_progress(env_name=env_name, filename=f, log_path=log_path)
    )

    for filename in sorted_by_training_progress:
        if filename.endswith(".json"):
            json_data = read_logs(log_path=log_path, filename=filename)
            training_logs = parse_training_logs(env_name=env_name, json_data=json_data)
            if not training_logs.is_agent_state_empty():
                exploration_coef = training_logs.get_exploration_coefficient()
                training_progress = training_logs.get_training_progress()
                if exploration_coef > max_exploration_coef:
                    max_exploration_coef = exploration_coef

                if training_progress_filter is None or training_progress >= training_progress_filter:
                    file_suffices.append(int(filename[filename.rindex("_") + 1 : filename.index(".")]))
                    data_point = Data(filename=filename, training_logs=training_logs,)
                    dataset.add(data_point)

                    if data_point.label == 1:
                        num_failures += 1

                    if save_data:
                        image_array = training_logs.get_image()
                        image = Image.fromarray(image_array.astype(np.uint8))
                        filepath = os.path.join(
                            log_path,
                            os.path.join(
                                "preprocessed",
                                "failed" if data_point.label == 1 else "succeeded",
                                "env-{}-{}-{}.tiff".format(count, data_point.label, round(training_progress, 0)),
                            ),
                        )

                        image.save(fp=filepath)
                        count += 1

    min_episode = np.min(file_suffices)
    max_episode = np.max(file_suffices)

    # TODO: refactor and do this in each training RL algo
    if max_exploration_coef > 0.0:
        for data_point in dataset.get():
            data_point.exploration_coef = round((data_point.exploration_coef / max_exploration_coef) * 100, 2)

    labels = [data.label for data in dataset.get()]
    logger.info("Number of samples: {}".format(len(labels)))
    logger.info("Data balancing: {}".format(np.mean(labels)))
    logger.info(
        "Training progress filter {}. Considering env configurations from episode {} to episode {}. "
        "Number of failures {}/{}".format(
            training_progress_filter, min_episode, max_episode, num_failures, max_episode - min_episode + 1
        )
    )

    return dataset
