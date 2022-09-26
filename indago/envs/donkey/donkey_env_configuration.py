import copy
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from indago.avf.env_configuration import EnvConfiguration, EnvMutations
from indago.config import COMMAND_SEPARATOR
from indago.envs.donkey.track_generator.track_elem import TrackElem
from indago.envs.donkey.track_generator.track_generator import TrackGenerator
from indago.envs.donkey.track_generator.unity.command import Command, parse_command
from indago.envs.donkey.track_generator.unity.utils import get_track_points
from indago.utils import randomness


class DonkeyEnvConfiguration(EnvConfiguration):
    def __init__(
        self, track_elements: List[TrackElem] = None,
    ):
        super().__init__()
        self.key_names = ["track"]
        self.track_elements = track_elements
        self.track_generator = TrackGenerator()

        # FIXME: in the other environments there is no check of validity when the configuration is instantiated
        # if track_elements is not None:
        #     assert self._is_valid(track_elements=track_elements), 'Track {} not valid'.format(self.get_str())

        self.update_implementation(track=self.track_elements,)

    def generate_configuration(self) -> "EnvConfiguration":

        track_elements = self.track_generator.generate()
        while not self._is_valid(track_elements=track_elements):
            track_elements = self.track_generator.generate()

        self.track_elements = copy.deepcopy(track_elements)
        self.update_implementation(track=self.track_elements,)

        return self

    def _is_valid(self, track_elements: List[TrackElem]) -> bool:
        return self.track_generator.constraints_satisfied(track_elements=track_elements)

    def get_length(self) -> int:
        assert len(self.track_elements) > 0, "Track elements still not initialized"
        return len(self.track_elements)

    @staticmethod
    # Region Of Interest
    # r = [margin_left, margin_top, width, height]
    def get_roi() -> List[int]:
        return [30, 25, 150, 150]

    def get_testing_image(self, car_trajectory: List[Tuple[float, float]]) -> np.ndarray:
        track_points = get_track_points(track_elements=self.track_elements)
        plot_x = [tp.x for tp in track_points]
        plot_y = [tp.y for tp in track_points]
        roi = self.get_roi()

        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        ax.axis("off")
        ax.scatter(plot_x, plot_y, c="black", s=50)

        # first components have some weird values, 49 and 50 are values of the starting position (x, y) that is
        # fixed, i.e. always the same for every track
        first_value_in = False
        car_trajectory_x, car_trajectory_y = [], []
        for idx, trajectory_component in enumerate(car_trajectory):
            if first_value_in:
                car_trajectory_x.append(trajectory_component[0])
                car_trajectory_y.append(trajectory_component[1])
            else:
                if (round(trajectory_component[0], 0) == 50.0 or round(trajectory_component[0], 0) == 49.0) and (
                    round(trajectory_component[1], 0) == 50.0 or round(trajectory_component[1], 0) == 49.0
                ):
                    first_value_in = True
        ax.scatter(car_trajectory_x, car_trajectory_y, c="red", s=50)

        canvas.draw()
        buf = canvas.buffer_rgba()
        image_array = np.asarray(buf)
        image_array = image_array.astype(np.float32)
        image_array = cv2.resize(image_array, (200, 200))
        # remove contours, including axes labels
        image_array = image_array[int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])]
        return image_array

    def get_image(self) -> np.ndarray:
        track_points = get_track_points(track_elements=self.track_elements)
        plot_x = [tp.x for tp in track_points]
        plot_y = [tp.y for tp in track_points]
        roi = self.get_roi()

        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        ax.axis("off")
        ax.plot(plot_x, plot_y, c="black", linewidth=20)

        canvas.draw()
        buf = canvas.buffer_rgba()
        image_array = np.asarray(buf)
        image_array = image_array.astype(np.float32)
        image_array = cv2.resize(image_array, (200, 200))

        # remove contours, including axes labels
        image_array = image_array[int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])]
        return image_array

    def get_str(self) -> str:
        str_list = []
        for ce in self.track_elements:
            str_list.append("{} {}".format(str(ce.command.name), str(ce.value)))
        return COMMAND_SEPARATOR.join(str_list)

    def str_to_config(self, s: str) -> "EnvConfiguration":
        split_instructions = s.split(COMMAND_SEPARATOR)
        track_elements: List[TrackElem] = []
        for split_instruction in split_instructions:
            split = None
            for c in Command:
                if c.name in split_instruction:
                    split = (split_instruction[: len(c.name)], split_instruction[len(c.name) : len(split_instructions)])
                    break
            assert split is not None, "No command found in {}".format(split_instruction)
            command_name = split[0]
            command_value = split[1]
            command, value = parse_command(command_name=command_name, command_value=command_value)
            track_elements.append(TrackElem(command=command, value=value))

        self.track_elements = track_elements
        self.update_implementation(track=self.track_elements,)

        return self

    def mutate(self) -> Optional["EnvConfiguration"]:
        new_env_config = copy.deepcopy(self)

        # FIXME: change only one parameter (i.e. either qpos or qvel) with equal probability

        for i in range(new_env_config.get_length()):
            if randomness.get_random_float(low=0, high=1) < 0.5:
                track_element = self.track_generator.change_command(idx=i, track_elements=new_env_config.track_elements)
            else:
                track_element = self.track_generator.change_value(idx=i, track_elements=new_env_config.track_elements)
            new_env_config.track_elements[i] = track_element

        if new_env_config._is_valid(track_elements=new_env_config.track_elements):
            return new_env_config

        return None

    def mutate_hot(self, attributions: np.ndarray, mapping: Dict) -> Optional["EnvConfiguration"]:
        new_env_config = copy.deepcopy(self)
        # get indices as if the array attributions was sorted and reverse it (::-1)
        # indices_sort = np.argsort(attributions)[::-1]
        idx_to_mutate = random.choices(population=list(range(0, len(attributions))), weights=np.abs(attributions), k=1)[0]
        keys_to_mutate = list(filter(lambda key: idx_to_mutate in mapping[key], mapping.keys()))
        assert len(keys_to_mutate) == 1, "There must be only one key where the attribution is max ({}). Found: {}".format(
            idx_to_mutate, len(keys_to_mutate)
        )
        key_to_mutate = keys_to_mutate[0]
        if np.all(attributions >= 0):
            sign = "rnd"
        elif attributions[idx_to_mutate] > 0:
            sign = "pos"
        elif attributions[idx_to_mutate] < 0:
            sign = "neg"
        else:
            sign = "rnd"
        idx_to_mutate_env_config = self.get_index_to_mutate_env_config(
            idx_to_mutate=idx_to_mutate, key_to_mutate=key_to_mutate, mapping=mapping
        )
        if key_to_mutate == "commands":
            track_element = self.track_generator.change_command(
                idx=idx_to_mutate_env_config, track_elements=new_env_config.track_elements, sign=sign
            )
        elif key_to_mutate == "values":
            track_element = self.track_generator.change_value(
                idx=idx_to_mutate_env_config, track_elements=new_env_config.track_elements, sign=sign
            )
        else:
            raise RuntimeError("Key not present {}".format(key_to_mutate))

        new_env_config.track_elements[idx_to_mutate_env_config] = track_element

        if new_env_config._is_valid(track_elements=new_env_config.track_elements):
            return new_env_config

        return None

    def crossover(self, other_env_config: "EnvConfiguration", pos1: int, pos2: int) -> Optional["EnvConfiguration"]:
        # FIXME: this crossover does not comply with the grammar of the track and it will likely result in
        #  an invalid configuration.
        new_env_config_impl = copy.deepcopy(self.impl)
        for i in range(pos1):
            new_env_config_impl["track"][i] = self.impl["track"][i]
        for i in range(pos2 + 1, self.get_length()):
            new_env_config_impl["track"][i] = other_env_config.impl["track"][i]

        new_env_config = DonkeyEnvConfiguration(track_elements=new_env_config_impl["track"])

        try:
            if new_env_config._is_valid(track_elements=new_env_config.track_elements):
                return new_env_config
        except AssertionError as ex:
            pass
        return None

    def get_index_to_mutate_env_config(self, idx_to_mutate: int, key_to_mutate: str, mapping: Dict) -> int:
        assert idx_to_mutate in mapping[key_to_mutate], "Index {} not in key {}".format(idx_to_mutate, key_to_mutate)
        if idx_to_mutate > len(self.track_elements) - 1:
            assert idx_to_mutate - len(self.track_elements) >= 0, "Negative index"
            return idx_to_mutate - len(self.track_elements)
        return idx_to_mutate
