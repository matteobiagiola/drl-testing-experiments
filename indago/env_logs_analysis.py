import argparse
import copy
import math
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from indago.avf.config import AVF_TRAIN_POLICIES
from indago.avf.dataset import Dataset
from indago.avf.env_configuration import EnvConfiguration
from indago.config import DONKEY_ENV_NAME, ENV_NAMES, HUMANOID_ENV_NAME, PARK_ENV_NAME
from indago.envs.donkey.donkey_dataset import DonkeyDataset
from indago.envs.donkey.donkey_env_configuration import DonkeyEnvConfiguration
from indago.envs.humanoid.humanoid_dataset import HumanoidDataset
from indago.envs.humanoid.humanoid_env_configuration import HumanoidEnvConfiguration
from indago.envs.park.parking_dataset import ParkingDataset
from indago.envs.park.parking_env_configuration import ParkingEnvConfiguration
from indago.stats.effect_size import cohend, odds_ratio_to_cohend, vargha_delaney
from indago.stats.power_analysis import fisher_power_analysis, parametric_power_analysis
from indago.stats.stat_tests import anova_plus_tukey, fisher_exact, mannwhitney_test
from log import Log

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Folder where logs are", type=str, required=True)
parser.add_argument(
    "--env-name", help="Env name", type=str, choices=ENV_NAMES, default=None
)
parser.add_argument(
    "--avf-train-policy",
    help="Avf train policy",
    type=str,
    choices=AVF_TRAIN_POLICIES,
    default="mlp",
)
parser.add_argument(
    "--alpha",
    help="Statistical significance level for statistical tests",
    type=float,
    default=0.05,
)
parser.add_argument(
    "--beta", help="Power level for statistical tests", type=float, default=0.8
)
parser.add_argument(
    "--no-adjust",
    help="Do not adjust p-values when multiple comparisons",
    action="store_false",
    default=True,
)
parser.add_argument(
    "--distance",
    help="Compute distance between env configs",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--files", nargs="+", help="List of files to analyze", required=True
)
parser.add_argument(
    "--names", nargs="+", help="Names associated to files", required=True
)
parser.add_argument(
    "--hist",
    help="Histogram plot instead of box plots for failure probability",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--hist-weighted",
    help="Number of failures weighed by failure probability (valid only for Humanoid and Donkey, i.e. non-deterministic environment)",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--times-elapsed",
    help="Box plot of times elapsed and statistical comparison",
    action="store_true",
    default=False,
)

args = parser.parse_args()


def check_duplicates(
    d: Dataset, config: EnvConfiguration, all_configs: List[EnvConfiguration]
) -> bool:
    for other_config in all_configs:
        if d.compute_distance(env_config_1=config, env_config_2=other_config) == 0:
            return True

    return False


if __name__ == "__main__":

    logger = Log("env_logs_analysis")
    logger.info("Args: {}".format(args))

    if args.distance:
        assert (
            args.env_name is not None
        ), "When distance flag is true env_name must be specified"

    times_elapsed_names = []
    failure_probabilities_names = []
    predictions_dict = dict()
    num_predictions_dict = dict()
    num_experiments = 0
    num_trials = 1
    env_configurations_dict = dict()
    str_env_configurations_dict = dict()
    indices_duplicated_configs_dict = dict()

    if args.env_name == PARK_ENV_NAME:
        dataset = ParkingDataset(policy=args.avf_train_policy)
    elif args.env_name == HUMANOID_ENV_NAME:
        dataset = HumanoidDataset(policy=args.avf_train_policy)
    elif args.env_name == DONKEY_ENV_NAME:
        dataset = DonkeyDataset(policy=args.avf_train_policy)
    else:
        raise NotImplementedError("Unknown env name: {}".format(args.env_name))

    for i in range(len(args.files)):
        assert os.path.exists(
            os.path.join(args.folder, args.files[i])
        ), "{} does not exist".format(os.path.join(args.folder, args.files[i]))
        with open(
            os.path.join(args.folder, args.files[i]), "r+", encoding="utf-8"
        ) as f:
            previous_line = None
            for line in f.readlines():
                if "INFO:Avf:Generating" in line:
                    sentence = line.split(":")[2]
                    num_trials = int("".join(list(filter(str.isdigit, sentence))))
                if "Times elapsed (s)" in line:
                    split = line.split(":")
                    string_nums = (
                        split[3]
                        .replace(" ", "")
                        .replace(",Mean", "")
                        .replace("[", "")
                        .replace("]", "")
                    )
                    times_elapsed = [
                        float(num_string) for num_string in string_nums.split(",")
                    ]
                    times_elapsed_names.append(times_elapsed)
                if "Failure probabilities" in line:
                    split = line.split(":")
                    string_nums = (
                        split[3]
                        .replace(" ", "")
                        .replace(",Mean", "")
                        .replace("[", "")
                        .replace("]", "")
                    )
                    failure_probabilities = [
                        float(num_string) for num_string in string_nums.split(",")
                    ]
                    failure_probabilities_names.append(failure_probabilities)
                if "Number of evaluation predictions" in line:
                    split = line.split(":")
                    num_predictions = int(split[3].replace(" ", "").replace("\n", ""))
                    num_predictions_dict[args.names[i]] = num_predictions
                if "Num experiments" in line:
                    # DEBUG:experiments:Num experiments: 1/200
                    split = line.split(":")
                    num_experiments = int(split[3].replace(" ", "").split("/")[1])
                if "max prediction: " in line:
                    prediction = float(
                        line[line.index("max prediction") :]
                        .split(":")[1]
                        .replace(" ", "")
                    )
                    if args.names[i] not in predictions_dict:
                        predictions_dict[args.names[i]] = []
                    predictions_dict[args.names[i]].append(prediction)
                if (
                    "INFO:Avf:Env configuration:" in line
                    and args.distance
                    and (
                        ";" in line
                        or args.names[i] == "random"
                        or "Fallback to random generation" in previous_line
                    )
                ):
                    split = line.split(":")
                    str_env_config = (
                        split[3].replace(" ", "").split(";")[0].replace("\n", "")
                    )
                    if args.names[i] == "random" and "/" in str_env_config:
                        continue

                    if args.names[i] not in env_configurations_dict:
                        env_configurations_dict[args.names[i]] = []

                    if args.names[i] not in str_env_configurations_dict:
                        str_env_configurations_dict[args.names[i]] = set()

                    # TODO: refactor
                    if args.env_name == PARK_ENV_NAME:
                        env_config = ParkingEnvConfiguration().str_to_config(
                            s=str_env_config
                        )
                    elif args.env_name == HUMANOID_ENV_NAME:
                        env_config = HumanoidEnvConfiguration().str_to_config(
                            s=str_env_config
                        )
                    elif args.env_name == DONKEY_ENV_NAME:
                        env_config = DonkeyEnvConfiguration().str_to_config(
                            s=str_env_config
                        )
                    else:
                        raise NotImplementedError(
                            "Unknown env name: {}".format(args.env_name)
                        )

                    env_configurations_dict[args.names[i]].append(env_config)

                    if args.names[i] not in indices_duplicated_configs_dict:
                        indices_duplicated_configs_dict[args.names[i]] = []

                    if str_env_config in str_env_configurations_dict[args.names[i]]:
                        if check_duplicates(
                            d=dataset,
                            config=env_config,
                            all_configs=env_configurations_dict[args.names[i]],
                        ):
                            # it is only for debugging purposes, duplicates have already been removed in the failure
                            # probabilities array
                            logger.warn(
                                "Duplicated configuration: {} in {}".format(
                                    str_env_config, args.names[i]
                                )
                            )
                            indices_duplicated_configs_dict[args.names[i]].append(
                                len(env_configurations_dict[args.names[i]])
                            )
                    str_env_configurations_dict[args.names[i]].add(str_env_config)

                previous_line = line

        if args.env_name == DONKEY_ENV_NAME and args.names[i] == "random":
            # exclude first env configuration needed to startup the donkey simulator but not used
            _ = env_configurations_dict["random"].pop(0)
        if args.distance:
            assert (
                len(env_configurations_dict[args.names[i]]) * num_trials
                == num_experiments
            ), "Num configurations {} != {} num experiments for method: {}".format(
                len(env_configurations_dict[args.names[i]]) * num_trials,
                num_experiments,
                args.names[i],
            )

    assert num_experiments != 0, "Num experiments cannot be 0"

    env_config_fp_pr = dict()
    pr_in_keys = len([key for key in env_configurations_dict.keys() if "pr" in key]) > 0
    if pr_in_keys:
        pr_key = [key for key in env_configurations_dict.keys() if "pr" in key]
        logger.info("Considering {} as pr for comparison".format(pr_key[0]))
        for i, name in enumerate(args.names):
            if name == pr_key[0]:
                failure_probabilities_pr = failure_probabilities_names[i]
                for j in range(len(failure_probabilities_pr)):
                    fp = failure_probabilities_pr[j]
                    env_config = env_configurations_dict[pr_key[0]][j]
                    env_config_fp_pr[env_config.get_str()] = fp

    # duplicates are not counted by default since failure probabilities are parsed from the end of the log file
    # where duplicates have been already removed
    for k in range(len(failure_probabilities_names)):
        fp = failure_probabilities_names[k]
        # it can happen when there are duplicates
        if len(fp) < num_experiments // num_trials:
            if args.hist:
                fp.extend([0] * (num_experiments // num_trials - len(fp)))
            else:
                fp.extend([0.0] * (num_experiments // num_trials - len(fp)))

    if args.hist:

        cp_failure_probabilities_names = copy.deepcopy(failure_probabilities_names)
        if args.env_name == DONKEY_ENV_NAME or args.env_name == HUMANOID_ENV_NAME:
            new_failure_probabilities_names = []
            for i in range(len(failure_probabilities_names)):
                fp = failure_probabilities_names[i]
                new_failure_probabilities_names.append(
                    list(map(lambda x: 1 if x > 0.5 else 0, fp))
                )
            failure_probabilities_names = new_failure_probabilities_names

        failure_names = []
        for i in range(len(args.files)):
            for is_failure in failure_probabilities_names[i]:
                if int(is_failure) == 1:
                    failure_names.append(args.names[i])

        # in case some techniques do not trigger any failure
        subtract_1 = False
        if len(failure_names) != len(args.names):
            subtract_1 = True
            for arg_name in args.names:
                failure_names.append(arg_name)

        df = pd.DataFrame({"Failures": failure_names})
        plt.figure()
        s = df["Failures"].value_counts()
        if subtract_1:
            s -= 1

        if args.hist_weighted:
            for i in range(len(args.names)):
                fps = np.asarray(cp_failure_probabilities_names[i])
                assert s[args.names[i]] == len(
                    fps[fps > 0.5]
                ), "Number of failures for {}, i.e. {} does not match the one computed from failure probabilities {}".format(
                    args.names[i], s[args.names[i]], len(fps[fps > 0.5])
                )

        s.plot(kind="barh")
        plt.title("# Failures in {} episodes (env configs)".format(num_experiments))

        # statistical tests
        if len(failure_probabilities_names) > 1:
            odds_ratios_p_values = fisher_exact(
                *failure_probabilities_names, adjust=not args.no_adjust
            )
            k = 0
            for i in range(len(args.files)):
                for j in range(i + 1, len(args.files)):
                    method_a = args.names[i]
                    method_b = args.names[j]
                    p_value = odds_ratios_p_values[k][1]
                    odds_ratio = odds_ratios_p_values[k][0]
                    effect_size, magnitude = odds_ratio_to_cohend(odds_ratio=odds_ratio)
                    failures_a = s[method_a]
                    failures_b = s[method_b]
                    if p_value < args.alpha:
                        if not args.no_adjust:
                            print(
                                "{} ({}) vs {} ({}) adjusted p-value: {}, odds ratio: {} (d: {}, {}), significant".format(
                                    method_a,
                                    s[method_a],
                                    method_b,
                                    s[method_b],
                                    p_value,
                                    odds_ratio,
                                    effect_size,
                                    magnitude,
                                )
                            )
                        else:
                            print(
                                "{} ({}) vs {} ({}) p-value: {}, odds ratio: {} (d: {}, {}), significant".format(
                                    method_a,
                                    s[method_a],
                                    method_b,
                                    s[method_b],
                                    p_value,
                                    odds_ratio,
                                    effect_size,
                                    magnitude,
                                )
                            )
                    else:
                        if failures_a / num_experiments == failures_b / num_experiments:
                            sample_size = math.inf
                        else:
                            sample_size = fisher_power_analysis(
                                p1=failures_a / num_experiments,
                                p2=failures_b / num_experiments,
                                power=args.beta,
                                sig=args.alpha,
                            )

                        if sample_size > num_experiments:
                            if not args.no_adjust:
                                print(
                                    "{} ({}) vs {} ({}) adjusted p-value: {}, odds ratio: {}, sample size: {}".format(
                                        method_a,
                                        s[method_a],
                                        method_b,
                                        s[method_b],
                                        p_value,
                                        odds_ratio,
                                        (
                                            int(sample_size)
                                            if sample_size != math.inf
                                            else math.inf
                                        ),
                                    )
                                )
                            else:
                                print(
                                    "{} ({}) vs {} ({}) p-value: {}, odds ratio: {}, sample size: {}".format(
                                        method_a,
                                        s[method_a],
                                        method_b,
                                        s[method_b],
                                        p_value,
                                        odds_ratio,
                                        (
                                            int(sample_size)
                                            if sample_size != math.inf
                                            else math.inf
                                        ),
                                    )
                                )
                        else:
                            if not args.no_adjust:
                                print(
                                    "{} ({}) vs {} ({}) adjusted p-value: {}, odds ratio: {}, not significant ({})".format(
                                        method_a,
                                        s[method_a],
                                        method_b,
                                        s[method_b],
                                        p_value,
                                        odds_ratio,
                                        (
                                            int(sample_size)
                                            if sample_size != math.inf
                                            else math.inf
                                        ),
                                    )
                                )
                            else:
                                print(
                                    "{} ({}) vs {} ({}) p-value: {}, odds ratio: {}, not significant ({})".format(
                                        method_a,
                                        s[method_a],
                                        method_b,
                                        s[method_b],
                                        p_value,
                                        odds_ratio,
                                        (
                                            int(sample_size)
                                            if sample_size != math.inf
                                            else math.inf
                                        ),
                                    )
                                )
                    k += 1
    else:
        to_boxplot_dict = dict()
        for i in range(len(args.files)):
            num_failures = sum(
                [
                    1 if failure_probability > 0.5 else 0
                    for failure_probability in failure_probabilities_names[i]
                ]
            )
            to_boxplot_dict[
                "{}_{}".format(args.names[i], num_failures)
            ] = failure_probabilities_names[i]

        plt.figure()
        plt.boxplot(
            x=to_boxplot_dict.values(),
            labels=to_boxplot_dict.keys(),
            showmeans=True,
            vert=False,
        )
        plt.title(
            "Failure probabilities in {} episodes (env configs)".format(num_experiments)
        )

        if args.adjust:
            # statistical tests with p-value adjustments
            tukey_result = anova_plus_tukey(
                lists=failure_probabilities_names, groups=args.names, alpha=args.alpha
            )
            if tukey_result is None:
                print("No statistical significance among the groups")
                # statistical tests
                if len(failure_probabilities_names) > 1:
                    for i in range(len(args.files)):
                        method_a = args.names[i]
                        num_failures_a = sum(
                            [
                                1 if failure_probability > 0.5 else 0
                                for failure_probability in failure_probabilities_names[
                                    i
                                ]
                            ]
                        )
                        for j in range(i + 1, len(args.files)):
                            num_failures_b = sum(
                                [
                                    1 if failure_probability > 0.5 else 0
                                    for failure_probability in failure_probabilities_names[
                                        j
                                    ]
                                ]
                            )
                            method_b = args.names[j]
                            effect_size, _ = cohend(
                                a=failure_probabilities_names[i],
                                b=failure_probabilities_names[j],
                            )
                            if math.isclose(effect_size, 0.0):
                                print(
                                    "{} ({}) vs {} ({}), not significant: effect size 0".format(
                                        method_a,
                                        num_failures_a,
                                        method_b,
                                        num_failures_b,
                                    )
                                )
                            else:
                                sample_size = parametric_power_analysis(
                                    effect=effect_size,
                                    alpha=args.alpha,
                                    power=args.beta,
                                )
                                if sample_size > num_experiments:
                                    print(
                                        "{} ({}) vs {} ({}), sample size: {}".format(
                                            method_a,
                                            num_failures_a,
                                            method_b,
                                            num_failures_b,
                                            (
                                                int(sample_size)
                                                if sample_size != math.inf
                                                else math.inf
                                            ),
                                        )
                                    )
                                else:
                                    print(
                                        "{} ({}) vs {} ({}), not significant ({})".format(
                                            method_a,
                                            num_failures_a,
                                            method_b,
                                            num_failures_b,
                                            (
                                                int(sample_size)
                                                if sample_size != math.inf
                                                else math.inf
                                            ),
                                        )
                                    )
            else:
                indices_names = dict()
                for i in range(len(tukey_result.groups)):
                    if tukey_result.groups[i] not in indices_names:
                        indices_names[tukey_result.groups[i]] = []
                    indices_names[tukey_result.groups[i]].append(i)
                for i in range(1, len(tukey_result._results_table)):
                    lst = tukey_result._results_table[i].data
                    method_a = lst[0]
                    method_b = lst[1]
                    p_value = lst[3]
                    failure_probabilities_a = tukey_result.data[
                        indices_names[method_a][0] : indices_names[method_a][-1]
                    ]
                    failure_probabilities_b = tukey_result.data[
                        indices_names[method_b][0] : indices_names[method_b][-1]
                    ]
                    num_failures_a = sum(
                        [
                            1 if failure_probability > 0.5 else 0
                            for failure_probability in failure_probabilities_a
                        ]
                    )
                    num_failures_b = sum(
                        [
                            1 if failure_probability > 0.5 else 0
                            for failure_probability in failure_probabilities_b
                        ]
                    )
                    if p_value < args.alpha:
                        eff_size_magnitude = vargha_delaney(
                            a=list(failure_probabilities_a),
                            b=list(failure_probabilities_b),
                        )
                        print(
                            "{} ({}) vs {} ({}), adjusted p-value: {}, effect size: {}, significant".format(
                                method_a,
                                num_failures_a,
                                method_b,
                                num_failures_b,
                                p_value,
                                eff_size_magnitude,
                            )
                        )
                    else:
                        effect_size, _ = cohend(
                            a=list(failure_probabilities_a),
                            b=list(failure_probabilities_b),
                        )
                        if math.isclose(effect_size, 0.0):
                            print(
                                "{} ({}) vs {} ({}), not significant: effect size 0".format(
                                    method_a, num_failures_a, method_b, num_failures_b
                                )
                            )
                        else:
                            sample_size = parametric_power_analysis(
                                effect=effect_size, alpha=args.alpha, power=args.beta
                            )
                            if sample_size > num_experiments:
                                print(
                                    "{} ({}) vs {} ({}), sample size: {}".format(
                                        method_a,
                                        num_failures_a,
                                        method_b,
                                        num_failures_b,
                                        (
                                            int(sample_size)
                                            if sample_size != math.inf
                                            else math.inf
                                        ),
                                    )
                                )
                            else:
                                print(
                                    "{} ({}) vs {} ({}), not significant ({})".format(
                                        method_a,
                                        num_failures_a,
                                        method_b,
                                        num_failures_b,
                                        (
                                            int(sample_size)
                                            if sample_size != math.inf
                                            else math.inf
                                        ),
                                    )
                                )
        else:
            for i in range(len(failure_probabilities_names)):
                fp_a = failure_probabilities_names[i]
                method_a = args.names[i]
                num_failures_a = sum(
                    [
                        1 if failure_probability > 0.5 else 0
                        for failure_probability in fp_a
                    ]
                )
                for j in range(i + 1, len(failure_probabilities_names)):
                    fp_b = failure_probabilities_names[j]
                    method_b = args.names[j]
                    num_failures_b = sum(
                        [
                            1 if failure_probability > 0.5 else 0
                            for failure_probability in fp_b
                        ]
                    )
                    _, p_value = mannwhitney_test(a=list(fp_a), b=list(fp_b))
                    if p_value < args.alpha:
                        eff_size_magnitude = vargha_delaney(a=list(fp_a), b=list(fp_b))
                        print(
                            "{} ({}) vs {} ({}), p-value: {}, effect size: {}, significant".format(
                                method_a,
                                num_failures_a,
                                method_b,
                                num_failures_b,
                                p_value,
                                eff_size_magnitude,
                            )
                        )
                    else:
                        effect_size, _ = cohend(a=list(fp_a), b=list(fp_b))
                        if math.isclose(effect_size, 0.0):
                            print(
                                "{} ({}) vs {} ({}), not significant: effect size 0".format(
                                    method_a, num_failures_a, method_b, num_failures_b
                                )
                            )
                        else:
                            sample_size = parametric_power_analysis(
                                effect=effect_size, alpha=args.alpha, power=args.beta
                            )
                            if sample_size > num_experiments:
                                print(
                                    "{} ({}) vs {} ({}), sample size: {}".format(
                                        method_a,
                                        num_failures_a,
                                        method_b,
                                        num_failures_b,
                                        (
                                            int(sample_size)
                                            if sample_size != math.inf
                                            else math.inf
                                        ),
                                    )
                                )
                            else:
                                print(
                                    "{} ({}) vs {} ({}), not significant ({})".format(
                                        method_a,
                                        num_failures_a,
                                        method_b,
                                        num_failures_b,
                                        (
                                            int(sample_size)
                                            if sample_size != math.inf
                                            else math.inf
                                        ),
                                    )
                                )

    if args.times_elapsed:
        to_boxplot_dict = dict()
        for i in range(len(args.files)):
            # remove first element which includes preprocessing of the dataset (outlier for some techniques)
            to_boxplot_dict[args.names[i]] = times_elapsed_names[i][1:]
        plt.figure()
        plt.boxplot(
            x=to_boxplot_dict.values(),
            labels=to_boxplot_dict.keys(),
            showmeans=True,
            vert=False,
        )
        plt.title("Times elapsed to generate env configuration")

    if len(predictions_dict.keys()) > 0:
        plt.figure()
        plt.boxplot(
            x=predictions_dict.values(),
            labels=predictions_dict.keys(),
            showmeans=True,
            vert=False,
        )
        plt.title("Max predictions")

    if args.distance:

        pr_in_keys = (
            len([key for key in env_configurations_dict.keys() if "pr" in key]) > 0
        )
        if pr_in_keys:
            pr_key = [key for key in env_configurations_dict.keys() if "pr" in key]
            logger.info("Considering {} as pr for comparison".format(pr_key[0]))
            env_configurations_pr = env_configurations_dict[pr_key[0]]
            distances_dict = dict()
            distances_zero_dict = dict()
            for name in args.names:
                if name != pr_key[0] and "pr" not in name:
                    env_configurations_other = env_configurations_dict[name]
                    for j in range(len(env_configurations_pr)):
                        env_configuration_pr = env_configurations_pr[j]
                        for k in range(j, len(env_configurations_other)):
                            env_configuration_other = env_configurations_other[k]
                            distance = dataset.compute_distance(
                                env_config_1=env_configuration_pr,
                                env_config_2=env_configuration_other,
                            )
                            if name not in distances_dict:
                                distances_dict[name] = []
                            if name not in distances_zero_dict:
                                distances_zero_dict[name] = 0
                            distances_dict[name].append(distance)
                            if distance == 0.0:
                                assert (
                                    env_configuration_pr.get_str() in env_config_fp_pr
                                ), "Env config {} is not in ".format(
                                    env_configuration_pr.get_str()
                                )
                                fp = env_config_fp_pr[env_configuration_pr.get_str()]
                                print(
                                    "Zero distance env: {}, index: {}, failure: {}".format(
                                        env_configuration_pr.get_str(), j, fp > 0.5
                                    )
                                )
                                distances_zero_dict[name] += 1

                    print(
                        "Distances PR vs {}. Mean: {}, Std: {}, Max: {}, Min: {}, Num zero distance: {}".format(
                            name,
                            np.mean(distances_dict[name]),
                            np.std(distances_dict[name]),
                            np.max(distances_dict[name]),
                            np.min(distances_dict[name]),
                            distances_zero_dict[name],
                        )
                    )

    print()

    if args.times_elapsed:
        print("Times elapsed statistical comparison")

        # remove first element which includes preprocessing of the dataset (outlier for some techniques)
        filtered_times_elapsed_names = []
        for i in range(len(times_elapsed_names)):
            filtered_times_elapsed_names.append(times_elapsed_names[i][1:])

        tukey_result = anova_plus_tukey(
            lists=filtered_times_elapsed_names, groups=args.names, alpha=args.alpha
        )
        if tukey_result is None:
            print("No statistical significance among the groups")
        else:
            indices_names = dict()
            for i in range(len(tukey_result.groups)):
                if tukey_result.groups[i] not in indices_names:
                    indices_names[tukey_result.groups[i]] = []
                indices_names[tukey_result.groups[i]].append(i)
            for i in range(1, len(tukey_result._results_table)):
                lst = tukey_result._results_table[i].data
                method_a = lst[0]
                method_b = lst[1]
                p_value = lst[3]
                times_elapsed_b = tukey_result.data[
                    indices_names[method_b][0] : indices_names[method_b][-1]
                ]
                times_elapsed_a = tukey_result.data[
                    indices_names[method_a][0] : indices_names[method_a][-1]
                ]
                if p_value < args.alpha:
                    eff_size_magnitude = vargha_delaney(
                        a=list(times_elapsed_a), b=list(times_elapsed_b)
                    )
                    print(
                        "{} ({}) vs {} ({}), adjusted p-value: {}, effect size: ({}, {}), significant".format(
                            method_a,
                            np.mean(times_elapsed_a),
                            method_b,
                            np.mean(times_elapsed_b),
                            p_value,
                            1 - eff_size_magnitude[0],
                            eff_size_magnitude[1],
                        )
                    )
                else:
                    effect_size, _ = cohend(
                        a=list(times_elapsed_a), b=list(times_elapsed_b)
                    )
                    if math.isclose(effect_size, 0.0):
                        print(
                            "{} ({}) vs {} ({}), not significant: effect size 0".format(
                                method_a,
                                np.mean(times_elapsed_a),
                                method_b,
                                np.mean(times_elapsed_b),
                            )
                        )
                    else:
                        sample_size = parametric_power_analysis(
                            effect=effect_size, alpha=args.alpha, power=args.beta
                        )
                        if sample_size > num_experiments:
                            print(
                                "{} ({}) vs {} ({}), sample size: {}".format(
                                    method_a,
                                    np.mean(times_elapsed_a),
                                    method_b,
                                    np.mean(times_elapsed_b),
                                    (
                                        int(sample_size)
                                        if sample_size != math.inf
                                        else math.inf
                                    ),
                                )
                            )
                        else:
                            print(
                                "{} ({}) vs {} ({}), not significant ({})".format(
                                    method_a,
                                    np.mean(times_elapsed_a),
                                    method_b,
                                    np.mean(times_elapsed_b),
                                    (
                                        int(sample_size)
                                        if sample_size != math.inf
                                        else math.inf
                                    ),
                                )
                            )

    plt.show()
