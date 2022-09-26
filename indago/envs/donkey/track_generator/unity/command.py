from enum import Enum
from typing import Tuple, Union


class Command(Enum):
    S = 0
    L = 1
    R = 2
    DY = 3


def parse_command(command_name: str, command_value: str) -> Tuple[Command, Union[float, int]]:
    if command_name == Command.S.name:
        command = Command.S
        value = int(command_value)
    elif command_name == Command.L.name:
        command = Command.L
        value = int(command_value)
    elif command_name == Command.R.name:
        command = Command.R
        value = int(command_value)
    elif command_name == Command.DY.name:
        command = Command.DY
        value = float(command_value)
    else:
        raise NotImplementedError("Unknown command {} {}".format(command_name, command_value))

    return command, value
