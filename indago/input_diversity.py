import copy
import glob
import os
from typing import Dict, List, Tuple

import numpy as np

from indago.avf.dataset import Dataset
from indago.avf.preprocessor import parse_testing_logs
from indago.utils.file_utils import read_logs
from log import Log


def input_diversity(
    folder: str, names: List[str], directories: List[str], env_name: str, dataset: Dataset, avf_train_policy: str
) -> Tuple[np.ndarray, Dict]:

    logger = Log("input_diversity")

    env_configurations_names = dict()
    for i in range(len(names)):
        exp_folder_name = directories[i]
        dirs = glob.glob(os.path.join(folder, exp_folder_name, "trial*"))

        env_configurations = []
        for dir_ in dirs:
            testing_logs_trial = None
            failure_flags = []
            for file in os.listdir(dir_):
                if ".json" in file:
                    failure_flags.append(int(file.split("-")[-1].split(".")[0]))
                    json_data = read_logs(log_path=dir_, filename=file)
                    testing_logs = parse_testing_logs(env_name=env_name, json_data=json_data)
                    testing_logs_trial = testing_logs
            assert testing_logs_trial is not None, "Testing logs not assigned in {} for {}".format(dir_, names[i])

            # # consider the configuration only if it is a failure
            # if sum(failure_flags) / len(failure_flags) > 0.5:

            # assumes the configuration triggers a failure (due to non-determinism this might not be true after
            # a re-execution)
            env_configurations.append(testing_logs_trial.config)

        env_configurations_names[names[i]] = copy.deepcopy(env_configurations)

    transformed_env_configurations_names = []
    for configs in env_configurations_names.values():
        transformed_env_configurations_names.extend(
            list(dataset.transform_env_configuration(env_configuration=config, policy=avf_train_policy)) for config in configs
        )

    logger.info("Transforming {} env configurations".format(len(transformed_env_configurations_names)))

    return np.asarray(transformed_env_configurations_names), env_configurations_names
