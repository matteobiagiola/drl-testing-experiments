import argparse
import glob
import math
import os

import numpy as np

from indago.avf.config import AVF_TRAIN_POLICIES
from indago.avf.env_configuration import EnvConfiguration
from indago.config import DONKEY_ENV_NAME, ENV_NAMES, HUMANOID_ENV_NAME, PARK_ENV_NAME
from indago.envs.donkey.donkey_env_configuration import DonkeyEnvConfiguration
from indago.envs.humanoid.humanoid_env_configuration import HumanoidEnvConfiguration
from indago.envs.park.parking_env_configuration import ParkingEnvConfiguration
from indago.stats.effect_size import cohend, vargha_delaney_unpaired
from indago.stats.power_analysis import parametric_power_analysis
from indago.stats.stat_tests import mannwhitney_test
from log import Log

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Folder where logs are", type=str, required=True)
parser.add_argument("--env-name", help="Env name", type=str, choices=ENV_NAMES, default=None)
parser.add_argument("--avf-train-policy", help="Avf train policy", type=str, choices=AVF_TRAIN_POLICIES, default="mlp")
parser.add_argument("--alpha", help="Statistical significance level for statistical tests", type=float, default=0.05)
parser.add_argument("--beta", help="Power level for statistical tests", type=float, default=0.8)
parser.add_argument("--adjust", help="Adjust p-values when multiple comparisons", action="store_true", default=False)
parser.add_argument("--names", nargs="+", help="Names associated to files", required=True)

args = parser.parse_args()


def get_env_config(s: str, env_name: str) -> EnvConfiguration:
    assert "FAIL -" in s, "String {} not supported".format(s)
    str_env_config = s.replace("INFO:experiments:FAIL - Failure probability for env config ", "").split(": ")[0]
    # TODO refactor
    if env_name == PARK_ENV_NAME:
        env_config_ = ParkingEnvConfiguration().str_to_config(s=str_env_config)
    elif env_name == HUMANOID_ENV_NAME:
        env_config_ = HumanoidEnvConfiguration().str_to_config(s=str_env_config)
    elif env_name == DONKEY_ENV_NAME:
        env_config_ = DonkeyEnvConfiguration().str_to_config(s=str_env_config)
    else:
        raise NotImplementedError("Unknown env name: {}".format(env_name))
    return env_config_


def get_failure_probability(s: str) -> float:
    assert "FAIL -" in s, "String {} not supported".format(s)
    return float(
        s.replace("INFO:experiments:FAIL - Failure probability for env config ", "")
        .split(": ")[1]
        .replace("(", "")
        .split(",")[0]
    )


if __name__ == "__main__":

    logger = Log("env_logs_analysis_trial")
    logger.info("Args: {}".format(args))

    failures_names = dict()
    for i in range(len(args.names)):
        name = args.names[i]
        if name == "random" or "prioritized_replay" in name:
            filenames = glob.glob(os.path.join(args.folder, "testing-{}-*-trial.txt".format(name)))
        else:
            filenames = glob.glob(os.path.join(args.folder, "testing-{}-{}-*-trial.txt".format(args.avf_train_policy, name)))
        assert len(filenames) > 0, "Log files for {} not found".format(name)
        for trial_num, filename in enumerate(filenames):
            assert os.path.exists(filename), "{} does not exist".format(filename)

            if name not in failures_names:
                failures_names[name] = dict()
            if trial_num not in failures_names[name]:
                failures_names[name][trial_num] = []

            with open(filename, "r+", encoding="utf-8") as f:
                for line in f.readlines():
                    if "INFO:experiments:FAIL -" in line:
                        env_config = get_env_config(s=line, env_name=args.env_name)
                        fp = get_failure_probability(s=line)
                        failures_names[name][trial_num].append((env_config, fp))

    lengths = [len(failures_names[name]) for name in failures_names.keys()]
    assert len(lengths) > 0 and sum(lengths) == lengths[0] * len(
        lengths
    ), "Number of trials must be the same for all names {}: {}".format(args.names, lengths)

    num_trials = lengths[0]
    if len(failures_names) == 1:
        # no statistical comparison
        num_failures = [len(failures_names[args.names[0]][key]) for key in failures_names[args.names[0]]]
        method = args.names[0]
        print("Failures {}: {}".format(method, num_failures))
    else:
        # statistical analysis (no adjust)
        for i in range(len(failures_names)):
            num_failures_a = [len(failures_names[args.names[i]][key]) for key in failures_names[args.names[i]]]
            method_a = args.names[i]
            for j in range(i + 1, len(failures_names)):
                num_failures_b = [len(failures_names[args.names[j]][key]) for key in failures_names[args.names[j]]]
                method_b = args.names[j]
                _, p_value = mannwhitney_test(a=list(num_failures_a), b=list(num_failures_b))
                print("Failures {}: {}, Failures {}: {}".format(method_a, num_failures_a, method_b, num_failures_b))
                if p_value < args.alpha:
                    eff_size_magnitude = vargha_delaney_unpaired(a=list(num_failures_a), b=list(num_failures_b))
                    print(
                        "{} ({}) vs {} ({}), p-value: {}, effect size: {}, significant".format(
                            method_a, np.mean(num_failures_a), method_b, np.mean(num_failures_b), p_value, eff_size_magnitude
                        )
                    )
                else:
                    effect_size, _ = cohend(a=list(num_failures_a), b=list(num_failures_b))
                    if math.isclose(effect_size, 0.0):
                        print(
                            "{} ({}) vs {} ({}), not significant".format(
                                method_a, np.mean(num_failures_a), method_b, np.mean(num_failures_b)
                            )
                        )
                    else:
                        sample_size = parametric_power_analysis(effect=effect_size, alpha=args.alpha, power=args.beta)
                        if sample_size > num_trials:
                            print(
                                "{} ({}) vs {} ({}), sample size: {}".format(
                                    method_a,
                                    np.mean(num_failures_a),
                                    method_b,
                                    np.mean(num_failures_b),
                                    int(sample_size) if sample_size != math.inf else math.inf,
                                )
                            )
                        else:
                            print(
                                "{} ({}) vs {} ({}), not significant ({})".format(
                                    method_a,
                                    num_failures_a,
                                    method_b,
                                    num_failures_b,
                                    int(sample_size) if sample_size != math.inf else math.inf,
                                )
                            )
