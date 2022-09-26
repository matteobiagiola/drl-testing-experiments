import math
from typing import List

import numpy as np
from euclid import Quaternion, Vector3

from indago.envs.donkey.track_generator.track_elem import TrackElem
from indago.envs.donkey.track_generator.unity.command import Command
from indago.envs.donkey.track_generator.unity.state import State
from indago.envs.donkey.track_generator.unity.track_point import TrackPoint
from indago.envs.donkey.track_generator.unity.track_script import TrackScript


def compute_distance(tp_1: TrackPoint, tp_2: TrackPoint) -> float:
    return math.hypot(tp_2.x - tp_1.x, tp_2.y - tp_1.y)


def has_loops(track_points: List[TrackPoint]) -> int:
    num_duplicates = 0
    for i in range(len(track_points)):
        tp_1 = track_points[i]
        for j in range(i + 1, len(track_points)):
            tp_2 = track_points[j]
            if math.isclose(abs(tp_1.x - tp_2.x), 0.0, abs_tol=1.0) and math.isclose(abs(tp_1.y - tp_2.y), 0.0, abs_tol=1.0):
                num_duplicates += 1
    return num_duplicates


def compute_direction(
    first_point: TrackPoint, second_point: TrackPoint, last_but_one_point: TrackPoint, last_point: TrackPoint
) -> float:
    v1 = np.asarray([second_point.x, second_point.y]) - np.asarray([first_point.x, first_point.y])
    v2 = np.asarray([last_point.x, last_point.y]) - np.asarray([last_but_one_point.x, last_but_one_point.y])
    unit1 = v1 / np.linalg.norm(v1)
    unit2 = v2 / np.linalg.norm(v2)
    dot_product = round(np.dot(unit1, unit2), 2)
    assert -1 <= dot_product <= 1, "Arccos not defined for the dot_product {}".format(dot_product)
    return np.degrees(np.arccos(dot_product))


def get_road_length(chromosome_elements: List[TrackElem]) -> float:
    length = 0
    for ce in chromosome_elements:
        if ce.command != Command.DY:
            length += ce.value
    return length


def track_closed(track_points: List[TrackPoint]) -> float:
    first_track_point = track_points[0]
    last_track_point = track_points[-1]
    return compute_distance(tp_1=first_track_point, tp_2=last_track_point)


def get_track_points(track_elements: List[TrackElem], consider_dy_segments: bool = False) -> List[TrackPoint]:
    """
    Copyright (c) 2017, Tawn Kramer
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of the <organization> nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    track_points: List[TrackPoint] = []

    start_position = Vector3(x=49.7, y=0.5, z=48.7)
    span = Vector3(x=0.0, y=0.0, z=0.0)
    span_dist = 2

    track_script = TrackScript(track_elements=track_elements)
    if track_script.parse_chromosome():

        dy = 0.0

        s = start_position
        s.y = 0.5
        span.x = 0.0
        span.y = 0.0
        span.z = span_dist
        turn_val = 10.0

        num_segments = 0

        for track_script_element in track_script.track:

            if track_script_element.state == State.AngleDY:
                turn_val = track_script_element.value
            elif track_script_element.state == State.CurveY:
                dy = track_script_element.value * turn_val
            else:
                dy = 0.0

            if consider_dy_segments and track_script_element.num_to_set == 0:
                track_points.append(TrackPoint(x=track_points[-1].x, y=track_points[-1].y, num_segment=num_segments))

            for i in range(track_script_element.num_to_set):
                track_points.append(TrackPoint(x=s.x, y=s.z, num_segment=num_segments))
                turn = dy
                rot = Quaternion.new_rotate_euler(math.radians(turn), 0, 0)
                span = rot * span.normalized()
                span *= span_dist
                s = s + span

            num_segments += 1

    return track_points
