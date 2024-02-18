import json
import os
from typing import List, Tuple, Union

from indago.avf.env_configuration import EnvConfiguration
from indago.config import DONKEY_ENV_NAME, ENV_NAMES, HUMANOID_ENV_NAME, PARK_ENV_NAME
from indago.envs.donkey.donkey_env_configuration import DonkeyEnvConfiguration
from indago.envs.humanoid.humanoid_env_configuration import HumanoidEnvConfiguration
from indago.envs.park.parking_env_configuration import ParkingEnvConfiguration


def read_logs(log_path: str, filename: str) -> json:
    assert os.path.exists(
        os.path.join(log_path, filename)
    ), "Filename {} not found".format(os.path.join(log_path, filename))
    with open(os.path.join(log_path, filename), "r+", encoding="utf-8") as f:
        return json.load(f)


def get_training_logs_path(
    folder: str, algo: str, env_id: str, exp_id: int, resume_dir: str = None
) -> str:
    if resume_dir is not None:
        log_path = os.path.join(folder, algo, resume_dir)
    else:
        log_path = os.path.join(folder, algo, "{}_{}".format(env_id, exp_id))
    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)
    return log_path


def parse_experiment_file(
    exp_file: str, env_name: str, return_failures: bool = True
) -> Union[List[EnvConfiguration], List[Tuple[EnvConfiguration, float]]]:
    assert os.path.exists(exp_file), "Exp file {} not found".format(exp_file)
    assert env_name in ENV_NAMES, "Env name {} not valid. Choose among: {}".format(
        env_name, ENV_NAMES
    )
    result = []
    f = False
    num_experiments = 0
    # for non deterministic environments
    num_trials = 1
    with open(exp_file, "r+", encoding="utf-8") as f:
        for line in f.readlines():

            if "Num experiments" in line:
                split = line.split(":")
                num_experiments = int(split[3].replace(" ", "").split("/")[1])

            if "INFO:Avf:Generating" in line:
                sentence = line.split(":")[2]
                num_trials = int("".join(list(filter(str.isdigit, sentence))))

            if return_failures and "INFO:experiments:FAIL -" in line:
                env_config_str = (
                    line.split(" - ")[1]
                    .split(":")[0]
                    .replace("Failure probability for env config ", "")
                )
                result.append(
                    get_env_config_of_env(
                        env_config_str=env_config_str, env_name=env_name
                    )
                )
            elif (
                not return_failures
                and f is True
                and "Failure probability for env config" in line
            ):
                env_config_str = (
                    line.split(":")[2]
                    .replace("FAIL - ", "")
                    .replace("Failure probability for env config ", "")
                )
                failure_probability = float(
                    line.split(":")[3].replace(" ", "").split("(")[1].split(",")[0]
                )
                env_config = get_env_config_of_env(
                    env_config_str=env_config_str, env_name=env_name
                )
                result.append((env_config, failure_probability))
            elif not return_failures and "DEBUG:experiments:Num experiments:" in line:
                current_episodes_count = int(
                    line.replace(" ", "").split(":")[3].split("/")[0]
                )
                total_episodes_count = int(
                    line.replace(" ", "").split(":")[3].split("/")[1]
                )
                if current_episodes_count == total_episodes_count:
                    f = True

    if return_failures:
        return result

    assert len(result) > 0, "No env configuration in exp file {}".format(exp_file)
    assert num_experiments > 0, "Num experiments cannot be <= 0"
    assert (
        len(result) == num_experiments // num_trials
    ), "Errors in parsing file. Num env configurations {} != num experiments {}".format(
        len(result), num_experiments // num_trials
    )

    return result


def get_env_config_of_env(env_config_str: str, env_name: str) -> EnvConfiguration:
    if env_name == PARK_ENV_NAME:
        return ParkingEnvConfiguration().str_to_config(s=env_config_str)
    elif env_name == HUMANOID_ENV_NAME:
        return HumanoidEnvConfiguration().str_to_config(s=env_config_str)
    elif env_name == DONKEY_ENV_NAME:
        return DonkeyEnvConfiguration().str_to_config(s=env_config_str)
    else:
        raise RuntimeError("Env name {} not supported".format(env_name))
