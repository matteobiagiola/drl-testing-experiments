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
from typing import List

from indago.envs.donkey.track_generator.track_elem import TrackElem
from indago.envs.donkey.track_generator.unity.command import Command
from indago.envs.donkey.track_generator.unity.state import State
from indago.envs.donkey.track_generator.unity.track_script_elem import TrackScriptElem


class TrackScript:
    def __init__(self, track_elements: List[TrackElem]):
        self.track: List[TrackScriptElem] = []
        self.track_elements = track_elements

    def parse_chromosome(self) -> bool:

        for ce in self.track_elements:

            command = ce.command
            args = ce.value

            tse = TrackScriptElem()

            if command.name == Command.S.name:
                tse.state = State.Straight
                tse.value = 1.0
                tse.num_to_set = int(args)
            elif command.name == Command.L.name:
                tse.state = State.CurveY
                tse.value = -1.0
                tse.num_to_set = int(args)
            elif command.name == Command.R.name:
                tse.state = State.CurveY
                tse.value = 1.0
                tse.num_to_set = int(args)
            elif command.name == Command.DY.name:
                tse.state = State.AngleDY
                tse.value = float(args)
                tse.num_to_set = 0
            else:
                raise NotImplementedError("Command {} not found".format(command.name))

            self.track.append(tse)

        return len(self.track) > 0
