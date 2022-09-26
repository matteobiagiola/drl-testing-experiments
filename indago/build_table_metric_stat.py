import argparse
import os
from typing import Tuple

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font

from log import Log

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Folder where logs are", type=str, default="logs")
parser.add_argument("--files", nargs="+", help="List of files to analyze", required=True)
parser.add_argument("--names", nargs="+", help="Names associated to files", required=True)
parser.add_argument(
    "--output-file",
    help="Name of the table file relative to folder",
    type=str,
    choices=[
        "failure_search",
        "input_diversity_coverage",
        "input_diversity_entropy",
        "output_diversity_coverage",
        "output_diversity_entropy",
    ],
    required=True,
)

args = parser.parse_args()


def prepend_suffix(s: str) -> str:
    """
    Only for results presentation
    """
    if s == "random":
        return "01_" + s
    if "pr" in s:
        return "02_" + s
    if "nn" in s:
        return "03_sampling"
    if "hc_rnd" == s or "hc" == s:
        return "04_" + s
    if "hc_fail" == s or "hc_failure" == s:
        return "05_" + s
    if "hc_sal_rnd" == s or "hc_saliency_rnd" == s:
        return "06_" + s
    if "hc_sal_fail" == s or "hc_saliency_failure" == s:
        return "07_" + s
    if "ga_rnd" == s:
        return "08_" + s
    if "ga_fail" == s or "ga_failure" == s:
        return "09_" + s
    if "ga_sal_rnd" == s or "ga_saliency_rnd" == s:
        return "10_" + s
    if "ga_sal_fail" == s or "ga_saliency_failure" == s:
        return "11_" + s
    raise NotImplementedError("Name {} not supported".format(s))


def get_name(s: str, left: bool = True) -> str:
    assert "vs" in s, "Not supported string: {}".format(s)
    if left:
        parsed_string = s.split(" vs ")[0].split(" (")[0]
    else:
        parsed_string = s.split(" vs ")[1].split(" (")[0]

    return prepend_suffix(s=parsed_string)


def get_metric_value(s: str, left: bool = True) -> float:
    assert "vs" in s, "Not supported string: {}".format(s)
    if left:
        return float(s[s.find("(") + 1 : s.find(")")])
    index_first_parenthesis = s.find("(")
    return float(s[s.find("(", index_first_parenthesis + 1) + 1 : s.find(")", s.find("vs"))])


def get_effect_size(s: str) -> Tuple[float, str]:
    assert "vs" in s, "Not supported string: {}".format(s)
    if "effect size" in s:
        index_first_parenthesis = s.find("(")
        index_second_parenthesis = s.find("(", index_first_parenthesis + 1)
        measure_magnitude_str = s[s.find("(", index_second_parenthesis + 1) + 1 : s.rfind(",") - 1]
        if "d: " in measure_magnitude_str:
            measure = float(measure_magnitude_str.split(",")[0].split(": ")[1])
        else:
            measure = float(measure_magnitude_str.split(",")[0])
        magnitude = measure_magnitude_str.split(", ")[1]
    elif "odds ratio" in s:
        measure = float(s.split(": ")[2].split(",")[0])
        magnitude = "n.a."
    else:
        measure = 0.0
        magnitude = "n.a."
    return measure, magnitude


def get_p_value(s: str) -> float:
    assert "vs" in s, "Not supported string: {}".format(s)
    if "p-value" in s or "sample size" in s:
        return float(s.split(": ")[1].split(",")[0])
    return -1.0


if __name__ == "__main__":

    logger = Log("build_table_failure_search")
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

    if args.output_file == "failure_search":
        headers = ["# Failures"]
    elif args.output_file == "input_diversity_coverage" or args.output_file == "output_diversity_coverage":
        headers = ["Coverage"]
    elif args.output_file == "input_diversity_entropy" or args.output_file == "output_diversity_entropy":
        headers = ["Entropy"]
    else:
        raise NotImplementedError("Output file {} not supported".format(args.output_file))

    failure_search_approaches = set()
    failure_search_metrics = dict()
    metrics_comparison = dict()

    times_elapsed_line = False
    diversity_flag = False

    for i in range(len(args.files)):
        assert os.path.join(args.files[i]), "{} does not exist".format(args.files[i])
        with open(args.files[i], "r+", encoding="utf-8") as f:
            for line in f.readlines():
                if "Times elapsed" in line:
                    times_elapsed_line = True

                if ("Coverage statistical analysis" in line and "entropy" in args.output_file) or (
                    "Entropy statistical analysis" in line and "coverage" in args.output_file
                ):
                    diversity_flag = True

                if ("Entropy statistical analysis" in line and "entropy" in args.output_file) or (
                    "Coverage statistical analysis" in line and "coverage" in args.output_file
                ):
                    diversity_flag = False

                if "vs" in line and "Distances" not in line and not times_elapsed_line and not diversity_flag:

                    # remove log suffix
                    line = line.replace("INFO:input_diversity:", "")
                    line = line.replace("INFO:output_diversity:", "")

                    print(line)

                    name_left = get_name(s=line)
                    name_right = get_name(s=line, left=False)

                    if name_left not in metrics_comparison:
                        metrics_comparison[name_left] = dict()
                        metrics_comparison[name_left][name_left] = ()

                    if name_left not in failure_search_metrics:
                        failure_search_metrics[name_left] = get_metric_value(s=line)

                    if name_right not in failure_search_metrics:
                        failure_search_metrics[name_right] = get_metric_value(s=line, left=False)

                    p_value = get_p_value(s=line)
                    effect_size_measure, effect_size_magnitude = get_effect_size(s=line)
                    metrics_comparison[name_left][name_right] = (p_value, effect_size_magnitude)

                    failure_search_approaches.add(name_left)
                    failure_search_approaches.add(name_right)

        print(sorted(failure_search_approaches))
        headers.extend(sorted(failure_search_approaches))

        workbook = Workbook()
        sheet = workbook.active

        for idx, header in enumerate(headers):
            sheet.column_dimensions[letters[idx]].width = 20
            if idx > 0:
                sheet["A{}".format(idx + 1)] = header

        assert len(headers) < len(letters), "Too many header entries (continue with AA, AB, AC...)"

        for idx in range(len(headers)):
            sheet["{}1".format(letters[idx + 1])] = headers[idx]

        for idx_1 in range(1, len(headers)):
            first_name = headers[idx_1]
            sheet["{}{}".format(letters[1], idx_1 + 1)] = failure_search_metrics[first_name]
            sheet["{}{}".format(letters[1], idx_1 + 1)].font = font
            sheet["{}{}".format(letters[1], idx_1 + 1)].alignment = central_alignment
            for idx_2 in range(idx_1, len(headers)):
                second_name = headers[idx_2]
                if first_name in metrics_comparison and second_name in metrics_comparison[first_name]:
                    p_value_effect_size = metrics_comparison[first_name][second_name]
                elif second_name in metrics_comparison and first_name in metrics_comparison[second_name]:
                    p_value_effect_size = metrics_comparison[second_name][first_name]
                elif first_name == second_name:
                    # the last failure search method is never a left name, therefore there is no comparison with itself
                    p_value_effect_size = ()
                else:
                    sheet["{}{}".format(letters[idx_1 + 1], idx_2 + 1)] = "xxxx"
                    sheet["{}{}".format(letters[idx_1 + 1], idx_2 + 1)].font = font
                    sheet["{}{}".format(letters[idx_1 + 1], idx_2 + 1)].alignment = central_alignment
                    print("Could not find match between {} and {}".format(first_name, second_name))
                    continue

                if len(p_value_effect_size) > 0:
                    p_value = str(round(p_value_effect_size[0], 3))
                    effect_size_magnitude = p_value_effect_size[1]
                    sheet["{}{}".format(letters[idx_1 + 1], idx_2 + 1)] = "{}, {}".format(p_value, effect_size_magnitude)
                else:
                    sheet["{}{}".format(letters[idx_1 + 1], idx_2 + 1)] = "----"

                sheet["{}{}".format(letters[idx_1 + 1], idx_2 + 1)].font = font
                sheet["{}{}".format(letters[idx_1 + 1], idx_2 + 1)].alignment = central_alignment

        workbook.save(filename="{}-{}.xlsx".format(os.path.join(args.folder, args.output_file), args.names[i]))

        failure_search_approaches.clear()
        failure_search_metrics.clear()
        metrics_comparison.clear()
        times_elapsed_line = False

        if args.output_file == "failure_search":
            headers = ["# Failures"]
        elif args.output_file == "input_diversity_coverage" or args.output_file == "output_diversity_coverage":
            headers = ["Coverage"]
        elif args.output_file == "input_diversity_entropy" or args.output_file == "output_diversity_entropy":
            headers = ["Entropy"]
        else:
            raise NotImplementedError("Output file {} not supported".format(args.output_file))
