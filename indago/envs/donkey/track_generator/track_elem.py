from typing import Union

from indago.envs.donkey.track_generator.unity.command import Command


class TrackElem:
    def __init__(self, command: Command, value: Union[float, int]):
        self.command = command
        self.value = value

    def __str__(self):
        return "({}, {})".format(self.command.name, self.value)
