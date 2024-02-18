import os
from typing import Tuple
import zipfile

import numpy as np
import json
from PIL import Image

from indago.avf.dataset import Data, Dataset
from indago.avf.testing_logs import TestingLogs
from indago.avf.training_logs import TrainingLogs
from indago.config import (
    DONKEY_ENV_NAME,
    ENV_NAMES,
    HUMANOID_ENV_NAME,
    PARK_ENV_NAME,
    CARTPOLE_ENV_NAME,
)
from indago.envs.cartpole.cartpole_dataset import CartPoleDataset
from indago.envs.cartpole.cartpole_env_configuration import CartPoleEnvConfiguration
from indago.envs.cartpole.cartpole_training_logs import CartPoleTrainingLogs
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
    if (
        env_name == PARK_ENV_NAME
        or env_name == HUMANOID_ENV_NAME
        or env_name == DONKEY_ENV_NAME
        or env_name == CARTPOLE_ENV_NAME
    ):
        env_config = json_data["env_config"]
        del json_data["env_config"]
        if json_data["agent_state"] is None:
            json_data["agent_state"] = dict()
        if env_name == PARK_ENV_NAME:
            return ParkingTrainingLogs(
                config=ParkingEnvConfiguration(**env_config), **json_data
            )
        elif env_name == HUMANOID_ENV_NAME:
            return HumanoidTrainingLogs(
                config=HumanoidEnvConfiguration(**env_config), **json_data
            )
        elif env_name == DONKEY_ENV_NAME:
            donkey_env_configuration = DonkeyEnvConfiguration()
            return DonkeyTrainingLogs(
                config=donkey_env_configuration.str_to_config(s=env_config), **json_data
            )
        elif env_name == CARTPOLE_ENV_NAME:
            return CartPoleTrainingLogs(
                config=CartPoleEnvConfiguration(**env_config), **json_data
            )
    else:
        raise NotImplementedError("Unknown env name: {}".format(env_name))


def parse_testing_logs(env_name: str, json_data) -> TestingLogs:
    if (
        env_name == PARK_ENV_NAME
        or env_name == HUMANOID_ENV_NAME
        or env_name == DONKEY_ENV_NAME
    ):
        env_config = json_data["env_config"]
        dynamic_info = json_data["dynamic_info"]
        # TODO: refactor
        if env_name == PARK_ENV_NAME:
            parking_env_configuration = ParkingEnvConfiguration()
            return TestingLogs(
                config=parking_env_configuration.str_to_config(s=env_config),
                dynamic_info=dynamic_info,
            )
        elif env_name == HUMANOID_ENV_NAME:
            humanoid_env_configuration = HumanoidEnvConfiguration()
            return TestingLogs(
                config=humanoid_env_configuration.str_to_config(s=env_config),
                dynamic_info=dynamic_info,
            )
        elif env_name == DONKEY_ENV_NAME:
            donkey_env_configuration = DonkeyEnvConfiguration()
            return TestingLogs(
                config=donkey_env_configuration.str_to_config(s=env_config),
                dynamic_info=dynamic_info,
            )
        elif env_name == CARTPOLE_ENV_NAME:
            cartpole_env_configuration = CartPoleEnvConfiguration()
            return TestingLogs(
                config=cartpole_env_configuration.str_to_config(s=env_config),
                dynamic_info=dynamic_info,
            )
    else:
        raise NotImplementedError("Unknown env name: {}".format(env_name))


def sort_training_progress(env_name: str, log_path: str, filename: str) -> float:
    if filename.endswith(".json"):
        json_data = read_logs(log_path=log_path, filename=filename)
        training_logs = parse_training_logs(env_name=env_name, json_data=json_data)
        return training_logs.get_training_progress()
    return 0.0


def preprocess_data(
    env_name: str,
    log_path: str,
    training_progress_filter: int,
    policy: str,
    save_data: bool = False,
    regression: bool = False,
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
    elif env_name == CARTPOLE_ENV_NAME:
        dataset = CartPoleDataset(policy=policy)
    else:
        raise NotImplementedError("Unknown env name: {}".format(env_name))

    # TODO: refactor and do this in each training RL algo
    # to estimate the max ent_coef (assuming this is the name of the exploration param in all algos)
    max_exploration_coef = 0.0
    count = 0
    num_failures = 0
    file_suffices = []
    all_training_logs = []

    if os.path.exists(os.path.join(log_path, "training_logs.zip")):
        with zipfile.ZipFile(
            os.path.join(log_path, "training_logs.zip"), mode="r"
        ) as archive:
            for info in archive.infolist():
                with archive.open(info.filename) as json_file:
                    json_data = json.load(json_file)
                    training_logs = parse_training_logs(
                        env_name=env_name, json_data=json_data
                    )
                    all_training_logs.append((training_logs, info.filename))

    if len(all_training_logs) > 0:
        sorted_by_training_progress = sorted(
            all_training_logs, key=lambda tl: tl[0].get_training_progress()
        )
    else:
        sorted_by_training_progress = sorted(
            os.listdir(path=log_path),
            key=lambda f: sort_training_progress(
                env_name=env_name, filename=f, log_path=log_path
            ),
        )

    for item in sorted_by_training_progress:
        if isinstance(item, Tuple):
            training_logs, filename = item
        else:
            filename = item
            if filename.endswith(".json"):
                json_data = read_logs(log_path=log_path, filename=item)
                training_logs = parse_training_logs(
                    env_name=env_name, json_data=json_data
                )
            else:
                training_logs = None

        if training_logs is not None and not training_logs.is_agent_state_empty():
            exploration_coef = training_logs.get_exploration_coefficient()
            training_progress = training_logs.get_training_progress()
            if exploration_coef > max_exploration_coef:
                max_exploration_coef = exploration_coef

            if (
                training_progress_filter is None
                or training_progress >= training_progress_filter
            ):
                file_suffices.append(
                    int(filename[filename.rindex("_") + 1 : filename.index(".")])
                )
                data_point = Data(
                    filename=filename,
                    training_logs=training_logs,
                    regression=regression,
                )
                dataset.add(data_point)

                # assuming fitness goes from 0 to 1
                if regression and data_point.regression_value == 0.0:
                    # label represents is_failure
                    assert (
                        data_point.label
                    ), f"Fitness value is zero but the agent succeeded. Inspect file {filename}. It could be that the fitness is zero because the simulator is flaky and it is assigning a zero fitness in a certain frame (which is not the final one). Remove the zero fitness if the last value of the fitness list is non-zero and re-execute."

                if data_point.label == 1:
                    num_failures += 1

                if save_data:
                    image_array = training_logs.get_image()
                    if image_array is not None:
                        image = Image.fromarray(image_array.astype(np.uint8))
                        filepath = os.path.join(
                            log_path,
                            os.path.join(
                                "preprocessed",
                                "failed" if data_point.label == 1 else "succeeded",
                                "env-{}-{}-{}.tiff".format(
                                    count, data_point.label, round(training_progress, 0)
                                ),
                            ),
                        )

                        image.save(fp=filepath)
                        count += 1

    min_episode = np.min(file_suffices)
    max_episode = np.max(file_suffices)

    # TODO: refactor and do this in each training RL algo
    if max_exploration_coef > 0.0:
        for data_point in dataset.get():
            data_point.exploration_coef = round(
                (data_point.exploration_coef / max_exploration_coef) * 100, 2
            )

    if regression:
        regression_values = np.asarray(
            [data.regression_value for data in dataset.get()]
        )

        max_regression_value = max(regression_values)

        near_failure_percentage = 10 * max_regression_value / 100

        # assuming they are fitness values
        failures = regression_values[regression_values == 0.0]
        near_failures = regression_values[
            (regression_values > 0.0) & (regression_values <= near_failure_percentage)
        ]
        other_values = regression_values[regression_values > near_failure_percentage]
        logger.info(f"Failures: {len(failures)}")
        logger.info(f"Near failures ({near_failure_percentage}): {len(near_failures)}")
        logger.info(f"Other values: {len(other_values)}")

        filenames_to_check = []
        for data_point in dataset.get():
            if (
                data_point.regression_value > 0
                and data_point.regression_value <= near_failure_percentage
            ):
                filenames_to_check.append(data_point.filename)

        if len(filenames_to_check) > 0:
            logger.info(
                f"Check these filenames {filenames_to_check} to make sure that the respective data points are actually near-failures and not the result of a flaky execution of the simulator (i.e., if the near-failure fitness values happens at the beginning of the episode)."
            )

    labels = [data.label for data in dataset.get()]
    logger.info("Number of samples: {}".format(len(labels)))
    logger.info("Data balancing: {}".format(np.mean(labels)))
    logger.info(
        "Training progress filter {}. Considering env configurations from episode {} to episode {}. "
        "Number of failures {}/{}".format(
            training_progress_filter,
            min_episode,
            max_episode,
            num_failures,
            max_episode - min_episode + 1,
        )
    )

    return dataset
