import copy
import math
from typing import List

from indago.config import (
    COMMAND_NAME_VALUE_DICT,
    MAX_DY_VALUE,
    MAX_VALUE_COMMANDS,
    MIN_VALUE_COMMANDS,
    NUM_COMMANDS_TRACK,
    TRACK_WIDTH,
)
from indago.envs.donkey.track_generator.track_elem import TrackElem
from indago.envs.donkey.track_generator.unity.command import Command
from indago.envs.donkey.track_generator.unity.utils import (
    compute_direction,
    get_track_points,
    has_loops,
    track_closed,
)
from indago.utils import randomness


class TrackGenerator:
    def __init__(self):
        self.min_value = MIN_VALUE_COMMANDS
        self.max_value = MAX_VALUE_COMMANDS
        self.length = NUM_COMMANDS_TRACK
        self.track_points = []
        self.track_elements = []

    def generate(self) -> List[TrackElem]:
        # S and L/R values must be integers; DY values can be floats
        result = [
            TrackElem(
                command=Command.S,
                value=randomness.get_random_int(
                    low=self.min_value, high=self.max_value
                ),
            )
        ]
        for i in range(self.length - 2):
            if result[-1].command.name == Command.DY.name:
                # L or R command after DY
                track_elem = TrackElem(
                    command=randomness.get_random_command(
                        excluded_commands=[Command.DY, Command.S]
                    ),
                    value=randomness.get_random_int(
                        low=self.min_value, high=self.max_value
                    ),
                )
            elif (
                result[-1].command.name == Command.R.name
                or result[-1].command.name == Command.L.name
                or result[-1].command.name == Command.S.name
            ) and i < self.length - 3:
                # S or DY command
                command = randomness.get_random_command(
                    excluded_commands=[Command.R, Command.L]
                )
                if command.name == Command.DY.name:
                    value = round(
                        randomness.get_random_float(
                            low=self.min_value, high=MAX_DY_VALUE
                        ),
                        1,
                    )
                else:
                    value = randomness.get_random_int(
                        low=self.min_value, high=self.max_value
                    )
                track_elem = TrackElem(command=command, value=value)
            else:
                track_elem = TrackElem(
                    Command.S,
                    value=randomness.get_random_int(
                        low=self.min_value, high=self.max_value
                    ),
                )

            result.append(track_elem)

        result.append(
            TrackElem(
                command=Command.S,
                value=randomness.get_random_int(
                    low=self.min_value, high=self.max_value
                ),
            )
        )
        return result

    @staticmethod
    def change_command(
        idx: int, track_elements: List[TrackElem], sign: str = "rnd"
    ) -> TrackElem:
        # FIXME: complicate this by analysing the previous command and the next one
        c_track_elements = copy.deepcopy(track_elements)
        track_element = c_track_elements[idx]
        new_te = TrackElem(command=track_element.command, value=track_element.value)

        if sign == "pos":
            if (
                new_te.command.name == Command.L.name
                and COMMAND_NAME_VALUE_DICT[Command.R.name]
                > COMMAND_NAME_VALUE_DICT[Command.L.name]
            ):
                new_te.command = Command.R
            elif (
                new_te.command.name == Command.R.name
                and COMMAND_NAME_VALUE_DICT[Command.L.name]
                > COMMAND_NAME_VALUE_DICT[Command.R.name]
            ):
                new_te.command = Command.L
        elif sign == "neg":
            if (
                new_te.command.name == Command.L.name
                and COMMAND_NAME_VALUE_DICT[Command.R.name]
                > COMMAND_NAME_VALUE_DICT[Command.L.name]
            ):
                new_te.command = Command.R
            elif (
                new_te.command.name == Command.R.name
                and COMMAND_NAME_VALUE_DICT[Command.L.name]
                > COMMAND_NAME_VALUE_DICT[Command.R.name]
            ):
                new_te.command = Command.L
        else:
            if new_te.command.name == Command.L.name:
                new_te.command = Command.R
            elif new_te.command.name == Command.R.name:
                new_te.command = Command.L

        return new_te

    def change_value(
        self, idx: int, track_elements: List[TrackElem], sign: str = "rnd"
    ) -> TrackElem:
        c_track_elements = copy.deepcopy(track_elements)
        track_element = c_track_elements[idx]
        new_te = TrackElem(command=track_element.command, value=track_element.value)
        max_value_dy = self.min_value + 1
        max_value_int = self.min_value + 1

        # S and R/L values must be integers; DY values can be floats
        if new_te.command.name == Command.DY.name:
            if sign == "pos":
                if new_te.value + 0.1 > MAX_DY_VALUE:
                    new_te.value = float(MAX_DY_VALUE)
                else:
                    v = round(
                        new_te.value
                        + round(
                            randomness.get_random_float(low=0, high=max_value_dy), 1
                        ),
                        1,
                    )
                    while v > MAX_DY_VALUE:
                        v = new_te.value
                        v = round(
                            v
                            + round(
                                randomness.get_random_float(low=0, high=max_value_dy), 1
                            ),
                            1,
                        )
                    new_te.value = v
            elif sign == "neg":
                if new_te.value - 0.1 < self.min_value:
                    new_te.value = float(self.min_value)
                else:
                    v = round(
                        new_te.value
                        - round(
                            randomness.get_random_float(low=0, high=max_value_dy), 2
                        ),
                        1,
                    )
                    while v < self.min_value:
                        v = new_te.value
                        v = round(
                            v
                            - round(
                                randomness.get_random_float(low=0, high=max_value_dy), 2
                            ),
                            1,
                        )
                    new_te.value = v
            else:
                if randomness.get_random_float(low=0, high=1) < 0.5:
                    if new_te.value + 0.1 > MAX_DY_VALUE:
                        new_te.value = float(MAX_DY_VALUE)
                    else:
                        v = round(
                            new_te.value
                            + round(
                                randomness.get_random_float(low=0, high=max_value_dy), 1
                            ),
                            1,
                        )
                        while v > MAX_DY_VALUE:
                            v = new_te.value
                            v = round(
                                v
                                + round(
                                    randomness.get_random_float(
                                        low=0, high=max_value_dy
                                    ),
                                    1,
                                ),
                                1,
                            )
                        new_te.value = v
                else:
                    if new_te.value - 0.1 < self.min_value:
                        new_te.value = float(self.min_value)
                    else:
                        v = round(
                            new_te.value
                            - round(
                                randomness.get_random_float(low=0, high=max_value_dy), 2
                            ),
                            1,
                        )
                        while v < self.min_value:
                            v = new_te.value
                            v = round(
                                v
                                - round(
                                    randomness.get_random_float(
                                        low=0, high=max_value_dy
                                    ),
                                    2,
                                ),
                                1,
                            )
                        new_te.value = v
        else:
            if sign == "pos":
                if new_te.value + self.min_value > self.max_value:
                    new_te.value = self.max_value
                else:
                    v = new_te.value + randomness.get_random_int(
                        low=self.min_value, high=max_value_int
                    )
                    while v > self.max_value:
                        v = new_te.value
                        v = v + randomness.get_random_int(
                            low=self.min_value, high=max_value_int
                        )
                    new_te.value = v
            elif sign == "neg":
                if new_te.value - self.min_value < self.min_value:
                    new_te.value = self.min_value
                else:
                    v = new_te.value - randomness.get_random_int(
                        low=self.min_value, high=max_value_int
                    )
                    while v < self.min_value:
                        v = new_te.value
                        v = v - randomness.get_random_int(
                            low=self.min_value, high=max_value_int
                        )
                    new_te.value = v
            else:
                if randomness.get_random_float(low=0, high=1) < 0.5:
                    if new_te.value + self.min_value > self.max_value:
                        new_te.value = self.max_value
                    else:
                        v = new_te.value + randomness.get_random_int(
                            low=self.min_value, high=max_value_int
                        )
                        while v > self.max_value:
                            v = new_te.value
                            v = v + randomness.get_random_int(
                                low=self.min_value, high=max_value_int
                            )
                        new_te.value = v
                else:
                    if new_te.value - self.min_value < self.min_value:
                        new_te.value = self.min_value
                    else:
                        v = new_te.value - randomness.get_random_int(
                            low=self.min_value, high=max_value_int
                        )
                        while v < self.min_value:
                            v = new_te.value
                            v = v - randomness.get_random_int(
                                low=self.min_value, high=max_value_int
                            )
                        new_te.value = v
        return new_te

    @staticmethod
    def is_track_containing_loops(track_elements: List[TrackElem]) -> bool:
        track_points = get_track_points(track_elements=track_elements)
        num_duplicates = has_loops(track_points=track_points)
        if num_duplicates == 1 and track_closed(track_points=track_points) < 1.0:
            return False
        return num_duplicates > 0

    def has_command_values_in_bounds(self, track_elements: List[TrackElem]) -> bool:
        for te in track_elements:
            if (
                te.command.name == Command.L.name
                or te.command.name == Command.R.name
                or te.command.name == Command.S.name
            ):
                if te.value < self.min_value or te.value > self.max_value:
                    return False
            elif te.command.name == Command.DY.name:
                if te.value < self.min_value or te.value > MAX_DY_VALUE:
                    return False
            else:
                raise NotImplementedError("Unknown command: {}".format(te.command))
        return True

    def functional_constraints(
        self, track_elements: List[TrackElem], remove_road_constraints: bool = False
    ) -> bool:
        in_bounds = self.has_command_values_in_bounds(track_elements=track_elements)
        no_loops = not self.is_track_containing_loops(track_elements=track_elements)
        no_sharp_turns = not self.has_very_sharp_turns(track_elements=track_elements)
        no_overlap = not self.has_overlapping_roads(track_elements=track_elements)
        if remove_road_constraints:
            return in_bounds and no_loops and no_sharp_turns and no_overlap
        no_straight_roads = not self.has_only_straight_roads(
            track_elements=track_elements
        )
        return (
            in_bounds
            and no_loops
            and no_sharp_turns
            and no_overlap
            and no_straight_roads
        )

    def constraints_satisfied(
        self, track_elements: List[TrackElem], remove_road_constraints: bool = False
    ) -> bool:
        functional_constraints = self.functional_constraints(
            track_elements=track_elements,
            remove_road_constraints=remove_road_constraints,
        )
        if remove_road_constraints:
            return functional_constraints

        difficult_curve = self.has_difficult_curve(
            track_elements=track_elements, curve_angle=130
        )
        two_curves = self.has_at_least_curves(
            track_elements=track_elements, num_curves=2
        )
        return functional_constraints and not difficult_curve and two_curves

    @staticmethod
    def has_at_least_curves(track_elements: List[TrackElem], num_curves: int) -> bool:
        num_curves_count = 0

        dy_command = None
        curve_command = None

        for i, ce in enumerate(track_elements):

            if ce.command.name == Command.DY.name:
                dy_command = i

            if ce.command.name == Command.R.name or ce.command.name == Command.L.name:
                curve_command = i

            if ce.command.name == Command.S.name and (dy_command and curve_command):
                num_curves_count += 1

                if num_curves_count == num_curves:
                    return True

                dy_command = None
                curve_command = None

        return False

    @staticmethod
    def has_length_between(
        track_elements: List[TrackElem], length_min: int, length_max: int
    ) -> bool:
        length_ = 0
        for ce in track_elements:
            if ce.command.name != Command.DY.name:
                length_ += ce.value
                if length_ > length_max:
                    return False
        return length_min <= length_ <= length_max

    @staticmethod
    def has_difficult_curve(track_elements: List[TrackElem], curve_angle: int) -> bool:
        previous_dy_value = None
        previous_curve_angle = 0
        for ce in track_elements:

            if ce.command.name == Command.DY.name:
                previous_dy_value = ce.value
            elif ce.command.name == Command.R.name:
                assert (
                    previous_dy_value is not None
                ), "Error assigning previous DY value"
                if previous_dy_value * ce.value >= curve_angle or (
                    previous_curve_angle + previous_dy_value * ce.value >= curve_angle
                ):
                    return True
                previous_curve_angle += previous_dy_value * ce.value
            elif ce.command.name == Command.L.name:
                assert (
                    previous_dy_value is not None
                ), "Error assigning previous DY value"
                if previous_dy_value * ce.value >= curve_angle or (
                    previous_curve_angle + previous_dy_value * ce.value >= curve_angle
                ):
                    return True
                previous_curve_angle += previous_dy_value * ce.value
            elif ce.command.name == Command.S.name:
                previous_curve_angle = 0
            else:
                raise NotImplementedError("Unknown command: {}".format(ce.command))

        return False

    @staticmethod
    def end_point_lower_than_start_point(track_elements: List[TrackElem]) -> bool:
        track_points = get_track_points(track_elements=track_elements)
        first_point = track_points[0]
        last_point = track_points[-1]
        return last_point.x > first_point.x and last_point.y < first_point.y

    @staticmethod
    def has_only_straight_roads(track_elements: List[TrackElem]) -> bool:
        for i in range(len(track_elements)):
            ce = track_elements[i]
            if ce.command.name != Command.S.name:
                return False
        return True

    @staticmethod
    def has_very_sharp_turns(track_elements: List[TrackElem]) -> bool:
        for i in range(len(track_elements)):
            ce = track_elements[i]
            if ce.command.name == Command.L.name or ce.command.name == Command.R.name:
                previous_ce = track_elements[i - 1]
                assert (
                    previous_ce.command.name == Command.DY.name
                ), "R/L command should be followed by DY. Found: {}".format(previous_ce)
                turn_angle = ce.value * previous_ce.value
                if turn_angle >= 170:
                    return True
        return False

    @staticmethod
    def has_overlapping_roads(track_elements: List[TrackElem]) -> bool:
        track_points = get_track_points(track_elements=track_elements)
        for i in range(len(track_points)):
            tp_i = track_points[i]
            # find point with the same y coordinate
            for j in range(i + 1, len(track_points)):
                tp_j = track_points[j]
                if (
                    abs(tp_i.num_segment - tp_j.num_segment) > 2
                    and math.isclose(a=tp_i.y, b=tp_j.y, abs_tol=1e-1)
                    and abs(tp_i.x - tp_j.x) < TRACK_WIDTH
                ):
                    return True

        for i in range(len(track_points)):
            tp_i = track_points[i]
            # find point with the same x coordinate
            for j in range(i + 1, len(track_points)):
                tp_j = track_points[j]
                if (
                    abs(tp_i.num_segment - tp_j.num_segment) > 2
                    and math.isclose(a=tp_i.x, b=tp_j.x, abs_tol=1e-1)
                    and abs(tp_i.y - tp_j.y) < TRACK_WIDTH
                ):
                    return True

        for i in range(0, len(track_points) - 1, 2):
            tp_i_1 = track_points[i]
            tp_i_2 = track_points[i + 1]
            for j in range(i + 2, len(track_points) - 1, 2):
                tp_j_1 = track_points[j]
                tp_j_2 = track_points[j + 1]
                if (
                    abs(tp_i_1.num_segment - tp_j_1.num_segment) > 2
                    and compute_direction(
                        first_point=tp_i_1,
                        second_point=tp_i_2,
                        last_point=tp_j_2,
                        last_but_one_point=tp_j_1,
                    )
                    > 150.0
                ):
                    if (
                        abs(tp_i_1.x - tp_j_1.x) < TRACK_WIDTH
                        or abs(tp_i_1.y - tp_j_1.y) < TRACK_WIDTH
                    ):
                        return True

        return False

    @staticmethod
    def is_there_any_negative_valued_command(track_elements: List[TrackElem]) -> bool:
        for ce in track_elements:
            if ce.value < 0:
                return True
        return False

    @staticmethod
    def road_condition(track_elements: List[TrackElem]) -> bool:
        road_condition = True
        for i in range(len(track_elements) - 1):
            ce_1 = track_elements[i]
            ce_2 = track_elements[i + 1]
            if ce_1.command.name == Command.DY.name:
                road_condition = (
                    ce_2.command.name == Command.R.name
                    or ce_2.command.name == Command.L.name
                )
                if not road_condition:
                    return False
            elif (
                ce_1.command.name == Command.R.name
                or ce_1 == Command.L.name
                or ce_1.command.name == Command.S.name
            ):
                road_condition = (
                    ce_2.command.name == Command.DY.name
                    or ce_2.command.name == Command.S.name
                )
                if not road_condition:
                    return False
            return road_condition

    def is_chromosome_correct(self, track_elements: List[TrackElem]) -> bool:
        length_condition = len(track_elements) == self.length
        first_ce = track_elements[0]
        first_command_condition = first_ce.command.name == Command.S.name
        last_ce = track_elements[-1]
        last_command_condition = last_ce.command.name == Command.S.name
        road_condition = self.road_condition(track_elements=track_elements)
        return (
            length_condition
            and first_command_condition
            and last_command_condition
            and road_condition
        )
