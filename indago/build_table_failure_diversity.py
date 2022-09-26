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
    "--type", help="Type of diversity to consider (input/output)", type=str, choices=["input", "output"], default="input"
)
parser.add_argument("--output-file", help="Name of the table file relative to folder", type=str, required=True)

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
    if "hc_rnd" == s:
        return "04_" + s
    if "hc_fail" == s:
        return "05_" + s
    if "hc_sal_rnd" == s:
        return "06_" + s
    if "hc_sal_fail" == s:
        return "07_" + s
    if "ga_rnd" == s:
        return "08_" + s
    if "ga_fail" == s:
        return "09_" + s
    if "ga_sal_rnd" == s:
        return "10_" + s
    if "ga_sal_fail" == s:
        return "11_" + s
    raise NotImplementedError("Name {} not supported".format(s))


def get_name(s: str) -> str:
    parsed_string = s.split(":")[2].split("method ")[1]
    return prepend_suffix(s=parsed_string)


def get_coverage(s: str) -> str:
    percentage = round(float(s.split(": ")[1].split("%")[0]), 2)
    return "{}".format(percentage)


def get_entropy(s: str) -> str:
    percentage = round(float(s.split(": ")[1].split("%")[0]), 2)
    return "{}".format(percentage)


if __name__ == "__main__":

    logger = Log("build_table_diversity")
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

    headers = []

    failure_search_metrics = dict()

    for i in range(len(args.files)):
        assert os.path.join(args.files[i]), "{} does not exist".format(args.files[i])
        with open(args.files[i], "r+", encoding="utf-8") as f:
            for line in f.readlines():

                if "Coverage" in line:
                    name = get_name(s=line)
                    if name not in failure_search_metrics:
                        failure_search_metrics[name] = []
                    failure_search_metrics[name].append(get_coverage(s=line))

                if "Entropy" in line and "Ideal" not in line:
                    failure_search_metrics[name].append(get_entropy(s=line))

        headers.extend(sorted(failure_search_metrics.keys()))

        workbook = Workbook()
        sheet = workbook.active

        for idx, header in enumerate(headers):
            sheet.column_dimensions[letters[idx]].width = 20
            sheet["A{}".format(idx + 2)] = header

        assert len(headers) < len(letters), "Too many header entries (continue with AA, AB, AC...)"

        sheet["{}1".format(letters[1])] = "Coverage (%)"
        sheet["{}1".format(letters[2])] = "Entropy (%)"

        for idx in range(len(headers)):
            name = headers[idx]
            sheet["{}{}".format(letters[1], idx + 2)] = failure_search_metrics[name][0]
            sheet["{}{}".format(letters[1], idx + 2)].font = font
            sheet["{}{}".format(letters[1], idx + 2)].alignment = central_alignment

            sheet["{}{}".format(letters[2], idx + 2)] = failure_search_metrics[name][1]
            sheet["{}{}".format(letters[2], idx + 2)].font = font
            sheet["{}{}".format(letters[2], idx + 2)].alignment = central_alignment

        workbook.save(filename="{}-{}-{}.xlsx".format(os.path.join(args.folder, args.output_file), args.names[i], args.type))

        failure_search_metrics.clear()
        headers.clear()
