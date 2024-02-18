import copy
import functools
import glob
import itertools
import os
from typing import Dict, List, Tuple

import numpy as np

from indago.avf.preprocessor import parse_testing_logs
from indago.avf.testing_logs import TestingLogs
from indago.utils.file_utils import read_logs
from log import Log


def max_length_testing_logs_fn(a: TestingLogs, b: TestingLogs) -> TestingLogs:
    if len(a.dynamic_info) > len(b.dynamic_info):
        return a
    return b


def max_length_dynamic_info_fn(a: List, b: List) -> List:
    if len(a) > len(b):
        return a
    return b


def flatten(lst: List) -> List:
    return list(itertools.chain(*lst))


def output_diversity(
    env_name: str, folder: str, names: List[str], directories: List[str]
) -> Tuple[np.ndarray, Dict]:

    logger = Log("output_diversity")
    env_configurations_names = dict()

    dynamic_infos_exp_names = dict()
    for i in range(len(names)):
        exp_folder_name = directories[i]
        dirs = glob.glob(os.path.join(folder, exp_folder_name, "trial*"))
        dirs = sorted(
            dirs, key=lambda filepath: int(filepath.split("/")[-1].split("-")[1])
        )
        dynamic_infos_exp = []
        env_configurations = []
        for dir_ in dirs:
            testing_logs_trial = []
            env_config = None
            # count the single failures within trial
            for file in os.listdir(dir_):
                if ".json" in file:
                    failure_flag = int(file.split("-")[-1].split(".")[0])
                    json_data = read_logs(log_path=dir_, filename=file)
                    testing_logs = parse_testing_logs(
                        env_name=env_name, json_data=json_data
                    )
                    if failure_flag:
                        testing_logs_trial.append(testing_logs)
                    env_config = testing_logs.config

            assert (
                env_config is not None
            ), "Env config not assigned in {} for {}".format(dir_, names[i])
            env_configurations.append(env_config)

            if len(testing_logs_trial) > 1:
                # within-trial padding
                testing_logs_max_length = functools.reduce(
                    max_length_testing_logs_fn, testing_logs_trial
                )
                padded_testing_logs_trial = []
                for testing_logs in testing_logs_trial:
                    if len(testing_logs.dynamic_info) < len(
                        testing_logs_max_length.dynamic_info
                    ):
                        diff = len(testing_logs_max_length.dynamic_info) - len(
                            testing_logs.dynamic_info
                        )
                        new_testing_logs = copy.deepcopy(testing_logs)
                        type_item = type(new_testing_logs.dynamic_info[0])
                        for k in range(diff):
                            if type_item == list:
                                new_testing_logs.dynamic_info.append(
                                    [
                                        0.0
                                        for _ in range(
                                            len(new_testing_logs.dynamic_info[0])
                                        )
                                    ]
                                )
                            elif type_item == float:
                                new_testing_logs.dynamic_info.append(0.0)
                            else:
                                raise NotImplementedError(
                                    "Type item {} not supported: {}".format(
                                        type_item, new_testing_logs.dynamic_info[0]
                                    )
                                )
                        assert len(new_testing_logs.dynamic_info) == len(
                            testing_logs_max_length.dynamic_info
                        ), "Error in padding {} != {}".format(
                            len(new_testing_logs.dynamic_info),
                            len(testing_logs_max_length.dynamic_info),
                        )
                        padded_testing_logs_trial.append(new_testing_logs)
                    elif len(testing_logs.dynamic_info) == len(
                        testing_logs_max_length.dynamic_info
                    ):
                        padded_testing_logs_trial.append(testing_logs)
                    else:
                        raise RuntimeError(
                            "Found a testing log whose dynamic info field {} is longer than the maximum {}".format(
                                len(testing_logs.dynamic_info),
                                len(testing_logs_max_length.dynamic_info),
                            )
                        )

                dynamic_infos = []
                dynamic_infos_item_length = 1
                for tl in padded_testing_logs_trial:
                    if type(tl.dynamic_info[0]) == list:
                        dynamic_infos.append(flatten(lst=tl.dynamic_info))
                        dynamic_infos_item_length = len(tl.dynamic_info[0])
                    elif type(tl.dynamic_info[0]) == float:
                        dynamic_infos.append(tl.dynamic_info)
                    else:
                        raise NotImplementedError(
                            "Type {} not supported".format(type(tl.dynamic_info[0]))
                        )

                shape = np.asarray(dynamic_infos).shape
                assert shape[0] == len(
                    testing_logs_trial
                ), "Number of rows in array must be = to {}. Found: {}".format(
                    len(testing_logs_trial), shape[0]
                )
                assert (
                    shape[1]
                    == len(testing_logs_max_length.dynamic_info)
                    * dynamic_infos_item_length
                ), "Number of columns in array must be = to {}. Found: {}".format(
                    len(testing_logs_max_length.dynamic_info)
                    * dynamic_infos_item_length,
                    shape[1],
                )

                mean_dynamic_info = np.mean(dynamic_infos, axis=0)
                assert (
                    mean_dynamic_info.shape[0]
                    == len(testing_logs_max_length.dynamic_info)
                    * dynamic_infos_item_length
                ), "Length of the array must be = to {}. Found: {}".format(
                    len(testing_logs_max_length.dynamic_info)
                    * dynamic_infos_item_length,
                    mean_dynamic_info.shape[0],
                )
                dynamic_infos_exp.append(list(mean_dynamic_info))
            elif len(testing_logs_trial) == 1:
                tl = testing_logs_trial[0]
                if type(tl.dynamic_info[0]) == list:
                    flattened = flatten(lst=tl.dynamic_info)
                    dynamic_infos_exp.append(flattened)
                elif type(tl.dynamic_info[0]) == float:
                    dynamic_infos_exp.append(tl.dynamic_info)
                else:
                    raise NotImplementedError(
                        "Type {} not supported".format(type(tl.dynamic_info[0]))
                    )
            else:
                raise RuntimeError(
                    "In directory {} no trials with failures. Remove directory and re-run.".format(
                        dir_
                    )
                )

        env_configurations_names[names[i]] = env_configurations
        dynamic_infos_exp_names[names[i]] = dynamic_infos_exp

    all_dynamic_infos_exp = []
    for dynamic_infos_exp in dynamic_infos_exp_names.values():
        all_dynamic_infos_exp.extend(dynamic_infos_exp)

    dynamic_infos_max_length = functools.reduce(
        max_length_dynamic_info_fn, all_dynamic_infos_exp
    )
    padded_dynamic_infos = []
    for dy_info in all_dynamic_infos_exp:
        if len(dy_info) < len(dynamic_infos_max_length):
            diff = len(dynamic_infos_max_length) - len(dy_info)
            new_dy_info = copy.deepcopy(dy_info)
            for k in range(diff):
                new_dy_info.append(0.0)
            assert len(new_dy_info) == len(
                dynamic_infos_max_length
            ), "Error in padding {} != {}".format(
                len(new_dy_info), len(dynamic_infos_max_length)
            )
            padded_dynamic_infos.append(new_dy_info)
        else:
            padded_dynamic_infos.append(dy_info)

    padded_dynamic_infos_np = np.asarray(padded_dynamic_infos)
    logger.info("Max length of dynamic infos: {}".format(len(dynamic_infos_max_length)))
    logger.info(
        "Shape of dynamic infos array: {}".format(padded_dynamic_infos_np.shape)
    )
    return padded_dynamic_infos_np, env_configurations_names
