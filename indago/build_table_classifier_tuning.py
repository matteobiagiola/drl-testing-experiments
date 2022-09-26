import argparse
import math
import os
from typing import Tuple

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font

from log import Log

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Folder where logs are", type=str, required=True)
parser.add_argument("--files", nargs="+", help="List of files to analyze", required=True)
parser.add_argument("--names", nargs="+", help="Names associated to files", required=True)
parser.add_argument("--table", help="Type of table to build", choices=["hyperparam"], type=str, required=True)
parser.add_argument("--output-file", help="Name of the table file relative to the folder", type=str, required=True)
parser.add_argument("--regression", help="Training performed with regression flag", action="store_true", default=False)

args = parser.parse_args()


def get_summary_hyperparam(s: str) -> Tuple[float, float, float, float]:
    return (
        float(s.split(":")[3].replace(" ", "").split(",")[0]),  # mean
        float(s.split(":")[4].replace(" ", "").split("(")[0]),  # std
        float(s.split(":")[5].replace(" ", "").split(",")[0]),  # min
        float(s.split(":")[6].replace(" ", "").split(",")[0]),  # max
    )


def get_summary_training_stability(s: str) -> float:
    return float(s.split(":")[3].replace(" ", ""))


def get_seed(s: str) -> int:
    return int(s.split(":")[2].split("seed")[1].replace(" ", ""))


def get_time_elapsed(s: str) -> float:
    return round(float(s.split(":")[3].replace(" ", "").replace("s", "")), 2)


if __name__ == "__main__":

    logger = Log("build_table")
    logger.info("Args: {}".format(args))

    assert len(args.files) == len(args.names), "Files length {} must be = to names length {}".format(
        len(args.files), len(args.names)
    )

    font = Font(name="Arial", size=10)
    central_alignment = Alignment(horizontal="center", vertical="center")

    letters = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]

    if args.table == "hyperparam":

        if not args.regression:
            headers = [
                "Test loss mean",
                "Test loss std",
                "Test precision mean",
                "Test precision std",
                "Test precision min",
                "Test recall mean",
                "Test recall std",
                "Test recall min",
                "F-measure",
                "Best epochs mean",
                "Best epochs std",
                "Best epochs min",
                "AUC mean",
                "Best (p, r, f, auc, seed)",
                "Best recall associated",
                "Time elapsed (s)",
            ]
        else:
            headers = [
                "Test loss mean",
                "Test loss std",
                "Test mae mean",
                "Test mae std",
                "Test mae min",
                "Test r2 mean",
                "Test r2 std",
                "Test r2 min",
                "Best epochs mean",
                "Best epochs std",
                "Best epochs min",
                "Best (mae, seed)",
                "Best r2 associated",
                "Time elapsed (s)",
            ]

        test_loss_names = dict()
        test_precision_names = dict()
        test_recall_names = dict()
        test_fmeasure_names = dict()
        test_epochs_names = dict()
        test_auc_names = dict()
        test_mae_names = dict()
        time_names = dict()
        test_r2_names = dict()

        seeds_names = dict()
        test_precisions_names = dict()
        test_recalls_names = dict()
        test_fmeasures_names = dict()
        test_aucs_names = dict()
        test_maes_names = dict()
        test_r2s_names = dict()

        for i in range(len(args.files)):
            assert os.path.join(args.folder, args.files[i]), "{} does not exist".format(
                os.path.join(args.folder, args.files[i])
            )
            with open(os.path.join(args.folder, args.files[i]), "r+", encoding="utf-8") as f:
                for line in f.readlines():
                    if "Training metrics" in line:
                        if "test losses" in line:
                            test_loss_names[args.names[i]] = get_summary_hyperparam(s=line)
                        elif "best epochs" in line:
                            test_epochs_names[args.names[i]] = get_summary_hyperparam(s=line)
                        elif "test precisions" in line:
                            test_precision_names[args.names[i]] = get_summary_hyperparam(s=line)
                        elif "test recalls" in line:
                            test_recall_names[args.names[i]] = get_summary_hyperparam(s=line)
                        elif "f-measures" in line:
                            test_fmeasure_names[args.names[i]] = get_summary_hyperparam(s=line)
                        elif "auc_rocs" in line:
                            test_auc_names[args.names[i]] = get_summary_hyperparam(s=line)
                        elif "test maes" in line:
                            test_mae_names[args.names[i]] = get_summary_hyperparam(s=line)
                        elif "test r2s" in line:
                            test_r2_names[args.names[i]] = get_summary_hyperparam(s=line)
                        else:
                            raise NotImplementedError("Unknown line: {}".format(line))
                    elif "Training stability run" in line:
                        if args.names[i] not in seeds_names:
                            seeds_names[args.names[i]] = []
                        seeds_names[args.names[i]].append(get_seed(s=line))
                    elif "Avf:Precision:" in line:
                        if args.names[i] not in test_precisions_names:
                            test_precisions_names[args.names[i]] = []
                        test_precisions_names[args.names[i]].append(get_summary_training_stability(s=line))
                    elif "Avf:Recall:" in line:
                        if args.names[i] not in test_recalls_names:
                            test_recalls_names[args.names[i]] = []
                        test_recalls_names[args.names[i]].append(get_summary_training_stability(s=line))
                    elif "Avf:F-measure:" in line:
                        if args.names[i] not in test_fmeasures_names:
                            test_fmeasures_names[args.names[i]] = []
                        test_fmeasures_names[args.names[i]].append(get_summary_training_stability(s=line))
                    elif "Avf:AUROC:" in line:
                        if args.names[i] not in test_aucs_names:
                            test_aucs_names[args.names[i]] = []
                        test_aucs_names[args.names[i]].append(get_summary_training_stability(s=line))
                    elif "Time elapsed" in line:
                        time_names[args.names[i]] = get_time_elapsed(s=line)
                    elif "INFO:Avf:MAE score" in line:
                        if args.names[i] not in test_maes_names:
                            test_maes_names[args.names[i]] = []
                        test_maes_names[args.names[i]].append(get_summary_training_stability(s=line))
                    elif "INFO:Avf:R2 score" in line:
                        if args.names[i] not in test_r2s_names:
                            test_r2s_names[args.names[i]] = []
                        test_r2s_names[args.names[i]].append(get_summary_training_stability(s=line))

        workbook = Workbook()
        sheet = workbook.active

        for i, name in enumerate(args.names):
            sheet["A{}".format(i + 2)] = name

        assert len(headers) < len(letters), "Too many header entries (continue with AA, AB, AC...)"

        for i in range(len(headers)):
            sheet["{}1".format(letters[i + 1])] = headers[i]

        for i, name in enumerate(args.names):

            sheet["{}{}".format(letters[1], i + 2)] = test_loss_names[name][0]
            sheet["{}{}".format(letters[1], i + 2)].font = font
            sheet["{}{}".format(letters[1], i + 2)].alignment = central_alignment

            sheet["{}{}".format(letters[2], i + 2)] = test_loss_names[name][1]
            sheet["{}{}".format(letters[2], i + 2)].font = font
            sheet["{}{}".format(letters[2], i + 2)].alignment = central_alignment

            if not args.regression:
                sheet["{}{}".format(letters[3], i + 2)] = test_precision_names[name][0]
                sheet["{}{}".format(letters[3], i + 2)].font = font
                sheet["{}{}".format(letters[3], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[4], i + 2)] = test_precision_names[name][1]
                sheet["{}{}".format(letters[4], i + 2)].font = font
                sheet["{}{}".format(letters[4], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[5], i + 2)] = test_precision_names[name][2]
                sheet["{}{}".format(letters[5], i + 2)].font = font
                sheet["{}{}".format(letters[5], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[6], i + 2)] = test_recall_names[name][0]
                sheet["{}{}".format(letters[6], i + 2)].font = font
                sheet["{}{}".format(letters[6], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[7], i + 2)] = test_recall_names[name][1]
                sheet["{}{}".format(letters[7], i + 2)].font = font
                sheet["{}{}".format(letters[7], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[8], i + 2)] = test_recall_names[name][2]
                sheet["{}{}".format(letters[8], i + 2)].font = font
                sheet["{}{}".format(letters[8], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[9], i + 2)] = test_fmeasure_names[name][0]
                sheet["{}{}".format(letters[9], i + 2)].font = font
                sheet["{}{}".format(letters[9], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[10], i + 2)] = test_epochs_names[name][0]
                sheet["{}{}".format(letters[10], i + 2)].font = font
                sheet["{}{}".format(letters[10], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[11], i + 2)] = test_epochs_names[name][1]
                sheet["{}{}".format(letters[11], i + 2)].font = font
                sheet["{}{}".format(letters[11], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[12], i + 2)] = test_epochs_names[name][2]
                sheet["{}{}".format(letters[12], i + 2)].font = font
                sheet["{}{}".format(letters[12], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[13], i + 2)] = test_auc_names[name][0]
                sheet["{}{}".format(letters[13], i + 2)].font = font
                sheet["{}{}".format(letters[13], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[13], i + 2)] = test_auc_names[name][0]
                sheet["{}{}".format(letters[13], i + 2)].font = font
                sheet["{}{}".format(letters[13], i + 2)].alignment = central_alignment

                # best_f_measure_index = np.argmax(test_fmeasures_names[name])
                best_precision_index = np.argmax(test_precisions_names[name])
                max_precision = test_precisions_names[name][best_precision_index]
                indices_with_max_precision = []
                print("Test precisions for {}: {}".format(name, test_precisions_names[name]))
                print("Test recalls for {}: {}".format(name, test_recalls_names[name]))
                for idx in range(len(test_precisions_names[name])):
                    if test_precisions_names[name][idx] == max_precision:
                        indices_with_max_precision.append(idx)
                if len(indices_with_max_precision) > 0:
                    index_with_max_recall = -1
                    max_recall = 0.0
                    for idx in indices_with_max_precision:
                        if test_recalls_names[name][idx] > max_recall:
                            max_recall = test_recalls_names[name][idx]
                            index_with_max_recall = idx
                    assert index_with_max_recall != -1, "Max recall index not assigned."
                    # take the index with the max recall if there are multiple indices with the same max precision
                    best_precision_index = index_with_max_recall

                print(
                    test_precisions_names[name],
                    test_precisions_names[name][best_precision_index],
                    test_recalls_names[name][best_precision_index],
                )
                sheet["{}{}".format(letters[14], i + 2)] = "{}-{}-{}-{}-{}".format(
                    test_precisions_names[name][best_precision_index],
                    test_recalls_names[name][best_precision_index],
                    test_fmeasures_names[name][best_precision_index],
                    test_aucs_names[name][best_precision_index],
                    seeds_names[name][best_precision_index],
                )
                sheet["{}{}".format(letters[14], i + 2)].font = font
                sheet["{}{}".format(letters[14], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[15], i + 2)] = test_recalls_names[name][best_precision_index]
                sheet["{}{}".format(letters[15], i + 2)].font = font
                sheet["{}{}".format(letters[15], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[16], i + 2)] = time_names[name]
                sheet["{}{}".format(letters[16], i + 2)].font = font
                sheet["{}{}".format(letters[16], i + 2)].alignment = central_alignment

                workbook.save(filename="{}.xlsx".format(os.path.join(args.folder, args.output_file)))
            else:

                sheet["{}{}".format(letters[3], i + 2)] = test_mae_names[name][0]
                sheet["{}{}".format(letters[3], i + 2)].font = font
                sheet["{}{}".format(letters[3], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[4], i + 2)] = test_mae_names[name][1]
                sheet["{}{}".format(letters[4], i + 2)].font = font
                sheet["{}{}".format(letters[4], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[5], i + 2)] = test_mae_names[name][2]
                sheet["{}{}".format(letters[5], i + 2)].font = font
                sheet["{}{}".format(letters[5], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[6], i + 2)] = test_r2_names[name][0]
                sheet["{}{}".format(letters[6], i + 2)].font = font
                sheet["{}{}".format(letters[6], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[7], i + 2)] = test_r2_names[name][1]
                sheet["{}{}".format(letters[7], i + 2)].font = font
                sheet["{}{}".format(letters[7], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[8], i + 2)] = test_r2_names[name][2]
                sheet["{}{}".format(letters[8], i + 2)].font = font
                sheet["{}{}".format(letters[8], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[9], i + 2)] = test_epochs_names[name][0]
                sheet["{}{}".format(letters[9], i + 2)].font = font
                sheet["{}{}".format(letters[9], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[10], i + 2)] = test_epochs_names[name][1]
                sheet["{}{}".format(letters[10], i + 2)].font = font
                sheet["{}{}".format(letters[10], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[11], i + 2)] = test_epochs_names[name][2]
                sheet["{}{}".format(letters[11], i + 2)].font = font
                sheet["{}{}".format(letters[11], i + 2)].alignment = central_alignment

                best_mae_index = np.argmin(test_maes_names[name])
                min_mae = test_maes_names[name][best_mae_index]

                indices_with_min_mae = []
                print("Test maes for {}: {}".format(name, test_maes_names[name]))
                print("Test r2s for {}: {}".format(name, test_r2s_names[name]))
                for idx in range(len(test_maes_names[name])):
                    if test_maes_names[name][idx] == min_mae:
                        indices_with_min_mae.append(idx)
                if len(indices_with_min_mae) > 0:
                    index_with_max_r2 = -1
                    max_r2 = -math.inf
                    for idx in indices_with_min_mae:

                        if test_r2s_names[name][idx] > max_r2:
                            max_r2 = test_maes_names[name][idx]
                            index_with_max_r2 = idx
                    assert index_with_max_r2 != -1, "Max r2 index not assigned."
                    # take the index with the max recall if there are multiple indices with the same max precision
                    best_mae_index = index_with_max_r2

                print(test_maes_names[name], test_maes_names[name][best_mae_index], test_r2s_names[name][best_mae_index])

                sheet["{}{}".format(letters[12], i + 2)] = "{}-{}".format(
                    test_maes_names[name][best_mae_index], seeds_names[name][best_mae_index]
                )
                sheet["{}{}".format(letters[12], i + 2)].font = font
                sheet["{}{}".format(letters[12], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[13], i + 2)] = test_r2s_names[name][best_mae_index]
                sheet["{}{}".format(letters[13], i + 2)].font = font
                sheet["{}{}".format(letters[13], i + 2)].alignment = central_alignment

                sheet["{}{}".format(letters[14], i + 2)] = time_names[name]
                sheet["{}{}".format(letters[14], i + 2)].font = font
                sheet["{}{}".format(letters[14], i + 2)].alignment = central_alignment

                workbook.save(filename="{}-rgr.xlsx".format(os.path.join(args.folder, args.output_file)))
    else:
        raise NotImplementedError("No support for table {}".format(args.table))
