# MIT License
#
# Copyright (c) 2018 Edouard Leurent
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import List, Tuple

import numpy as np
from gym import GoalEnv
from highway_env.envs import Action
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import MultiAgentObservation
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.objects import Landmark


class ParkingEnv(AbstractEnv, GoalEnv):
    """
    A continuous control environment.

    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to reach a given goal.

    Credits to Munir Jojo-Verge for the idea and initial implementation.
    """

    def _cost(self, action: Action) -> float:
        raise NotImplementedError("Not implemented")

    def __init__(
        self,
        num_lanes: int = None,
        goal_lane_idx: int = None,
        heading_ego: float = None,
        position_ego: Tuple[float, float] = None,
        parked_vehicles_lane_indices: List[int] = None,
    ):
        self.num_lanes = num_lanes
        self.goal_lane_idx = goal_lane_idx
        self.heading_ego = heading_ego
        self.position_ego = position_ego
        self.parked_vehicles_lane_indices = parked_vehicles_lane_indices
        self.parked_vehicles_positions = []
        self.distances_from_goals = []

        self.goal_pos = None
        # the following call creates the environment
        super(ParkingEnv, self).__init__()

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "KinematicsGoal",
                    "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "scales": [100, 100, 5, 5, 1, 1],
                    "normalize": False,
                },
                "action": {"type": "ContinuousAction"},
                "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
                "success_goal_reward": 0.12,
                "collision_reward": -5,
                "steering_range": np.deg2rad(45),
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 100,
                "screen_width": 600,
                "screen_height": 300,
                "centering_position": [0.5, 0.5],
                "scaling": 7,
                "controlled_vehicles": 1,
            }
        )
        return config

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminal, info = super().step(action)
        info.update(
            {
                "is_success": (
                    1
                    if self._is_success(obs["achieved_goal"], obs["desired_goal"])
                    else 0
                )
            }
        )

        # fitness length of the episode
        if terminal:
            info["fitness"] = 1 - self.steps / self.config["duration"]
        else:
            info["fitness"] = None

        return obs, reward, terminal, info

    def _info(self, obs, action) -> dict:
        info = super(ParkingEnv, self)._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(
                self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
                for agent_obs in obs
            )
        else:
            success = self._is_success(obs["achieved_goal"], obs["desired_goal"])
        info.update({"is_success": success})
        info.update({"vehicle_position": self.vehicle.position})
        info.update({"goal_position": self.goal.position})
        info.update({"parked_vehicles_positions": self.parked_vehicles_positions})
        return info

    def _reset(self):
        self.parked_vehicles_positions.clear()
        self._create_road()
        self._create_vehicles()

    def _create_road(self, spots: int = 15) -> None:
        """
        Create a road composed of straight adjacent lanes.

        :param spots: number of spots in the parking
        """
        net = RoadNetwork()
        width = 4.0
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 10
        length = 8
        if self.num_lanes is not None:
            spots = self.num_lanes
        for k in range(spots):
            x = (k - spots // 2) * (width + x_offset) - width / 2
            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [x, y_offset], [x, y_offset + length], width=width, line_types=lt
                ),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [x, -y_offset], [x, -y_offset - length], width=width, line_types=lt
                ),
            )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            if self.heading_ego is not None:
                if self.position_ego is not None:
                    vehicle = self.action_type.vehicle_class(
                        self.road,
                        [self.position_ego[0], self.position_ego[1]],
                        2 * np.pi * self.heading_ego,
                        0,
                    )
                else:
                    vehicle = self.action_type.vehicle_class(
                        self.road, [i * 20, 0], 2 * np.pi * self.heading_ego, 0
                    )
            else:
                if self.position_ego is not None:
                    vehicle = self.action_type.vehicle_class(
                        self.road,
                        [self.position_ego[0], self.position_ego[1]],
                        2 * np.pi * self.np_random.rand(),
                        0,
                    )
                else:
                    vehicle = self.action_type.vehicle_class(
                        self.road, [i * 20, 0], 2 * np.pi * self.np_random.rand(), 0
                    )
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)

        if self.parked_vehicles_lane_indices is not None:
            for i in range(len(self.parked_vehicles_lane_indices)):
                lane_idx = self.parked_vehicles_lane_indices[i]
                assert (
                    0 <= lane_idx < len(self.road.network.lanes_list())
                ), "Invalid lane num"
                assert (
                    lane_idx != self.goal_lane_idx
                ), "Parked vehicle cannot be in the target position"
                lane = self.road.network.lanes_list()[lane_idx]
                vehicle = self.action_type.vehicle_class(
                    road=self.road,
                    position=lane.position(lane.length / 2, 0),
                    heading=lane.heading,
                    speed=0,
                )
                self.parked_vehicles_positions.append(lane.position(lane.length / 2, 0))
                self.road.vehicles.append(vehicle)

        if self.goal_lane_idx is not None:
            assert (
                0 <= self.goal_lane_idx < len(self.road.network.lanes_list())
            ), "Invalid lane num"
            lane = self.road.network.lanes_list()[self.goal_lane_idx]
        else:
            lane = self.np_random.choice(self.road.network.lanes_list())

        self.goal = Landmark(
            self.road, lane.position(lane.length / 2, 0), heading=lane.heading
        )
        self.road.objects.append(self.goal)

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
        p: float = 0.5,
    ) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return (
            -np.power(
                np.dot(
                    np.abs(achieved_goal - desired_goal),
                    np.array(self.config["reward_weights"]),
                ),
                p,
            )
            + self.config["collision_reward"] * self.vehicle.crashed
        )

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        return sum(
            self.compute_reward(
                agent_obs["achieved_goal"], agent_obs["desired_goal"], {}
            )
            for agent_obs in obs
        )

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return (
            self.compute_reward(achieved_goal, desired_goal, {})
            > -self.config["success_goal_reward"]
        )

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached."""
        time = self.steps >= self.config["duration"]
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(
            self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
            for agent_obs in obs
        )
        return time or crashed or success
