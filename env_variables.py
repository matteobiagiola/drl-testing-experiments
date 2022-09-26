import sys


def parse_at_prefixed_params(args, param_name):
    try:
        index = args.index("--" + param_name)
        return args[index + 1]
    except ValueError:
        return None


home_abs_path = parse_at_prefixed_params(args=sys.argv, param_name="home_abs_path")
logging_level = parse_at_prefixed_params(args=sys.argv, param_name="logging_level")
num_of_threads = parse_at_prefixed_params(args=sys.argv, param_name="num_of_threads")

HOME = home_abs_path if home_abs_path else ".."
LOGGING_LEVEL = logging_level.upper() if logging_level else "DEBUG"
assert LOGGING_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
