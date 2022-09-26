import logging
import sys

from env_variables import LOGGING_LEVEL

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def close_loggers() -> None:
    # Remove all handlers associated with the root logger object. Needed to call logging.basicConfig multiple times
    # such that for different experiment runs a new log file is written
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


class Log:
    def __init__(self, logger_prefix: str, static_logger: bool = True) -> None:
        self.logger = logging.getLogger(logger_prefix)
        # avoid creating another logger if it already exists
        if len(self.logger.handlers) == 0 or not static_logger:
            self.logger = logging.getLogger(logger_prefix)
            self.logger.setLevel(level=LOGGING_LEVEL)
            formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            ch.setLevel(level=logging.DEBUG)

            self.logger.addHandler(ch)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warn(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
