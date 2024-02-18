"""
MIT License

Copyright (c) 2018 Roma Sokolkov
Copyright (c) 2018 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Original author: Tawn Kramer
import base64
import copy
import time
from io import BytesIO
from multiprocessing import Queue
from queue import Empty
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from indago.avf.avf import Avf
from indago.avf.env_configuration import EnvConfiguration
from indago.config import (
    CRASH_SPEED_WEIGHT,
    INPUT_DIM,
    MAX_CTE_ERROR,
    MAX_THROTTLE,
    MIN_THROTTLE,
    REWARD_CRASH,
    ROI,
)
from indago.envs.donkey.core.fps import FPSTimer
from indago.envs.donkey.core.message import IMesgHandler
from indago.envs.donkey.core.sim_client import SimClient
from indago.envs.donkey.donkey_training_logs import DonkeyTrainingLogs
from indago.envs.donkey.scenes.simulator_scenes import SimulatorScene
from indago.envs.donkey.vae.vae import VAE
from log import Log


class DonkeyUnitySimController:
    """
    Wrapper for communicating with unity simulation.
    """

    def __init__(
        self,
        seed: int,
        vae: VAE,
        port: int,
        socket_local_address: int,
        avf: Avf,
        simulation_mul: int,
        simulator_scene: SimulatorScene,
    ):
        self.port = port
        # sensor size - height, width, depth
        self.camera_img_size = INPUT_DIM

        self.address = ("127.0.0.1", port)
        self.socket_local_address = socket_local_address

        # Socket message handler
        self.handler = DonkeyUnitySimHandler(
            avf=avf,
            seed=seed,
            simulation_mul=simulation_mul,
            vae=vae,
            simulator_scene=simulator_scene,
        )

        self.client = SimClient(self.address, self.socket_local_address, self.handler)
        self.logger = Log("DonkeyUnitySimController")

    def close_connection(self):
        return self.client.handle_close()

    def wait_until_loaded(self):
        """
        Wait for a client (Unity simulator).
        """
        sleep_time = 0
        while not self.handler.loaded:
            time.sleep(0.1)
            sleep_time += 0.1
            if sleep_time > 3:
                self.logger.info(
                    "Waiting for sim to start..."
                    "if the simulation is running, press EXIT to go back to the menu"
                )
        self.regen_track()

    def reset(self, end_of_episode: bool = False):
        self.handler.reset(end_of_episode=end_of_episode)

    def regen_track(self):
        self.handler.generate_track()

    def seed(self, seed):
        self.handler.seed = seed

    def get_sensor_size(self):
        """
        :return: (int, int, int)
        """
        return self.handler.get_sensor_size()

    def take_action(self, action):
        self.handler.take_action(action)

    def observe(self):
        """
        :return: (np.ndarray)
        """
        return self.handler.observe()

    def quit(self):
        self.logger.info("Stopping client")
        self.client.stop()

    def render(self, mode):
        pass

    def is_game_over(self):
        return self.handler.is_game_over()

    def calc_reward(self, done):
        return self.handler.calc_reward(done)

    def send_agent_state(self, agent_state: Dict):
        self.handler.send_agent_state(agent_state=agent_state)


class DonkeyUnitySimHandler(IMesgHandler):
    """
    Socket message handler.
    """

    def __init__(
        self,
        seed: int,
        vae: VAE,
        avf: Avf,
        simulation_mul: int,
        simulator_scene: SimulatorScene,
    ):

        self.logger = Log("DonkeyUnitySimHandler")
        self.vae = vae
        self.simulation_mul = simulation_mul
        self.avf = avf
        self.is_success = 0

        self.loaded = False
        self.control_timer = FPSTimer(timer_name="control", verbose=0)
        self.observation_timer = FPSTimer(timer_name="observation", verbose=0)
        self.max_cte_error = MAX_CTE_ERROR
        self.seed = seed
        self.simulator_scene = simulator_scene

        # sensor size - height, width, depth
        self.camera_img_size = INPUT_DIM
        self.image_array = np.zeros(self.camera_img_size)
        self.original_image = None
        self.last_obs = None
        self.last_throttle = 0.0
        # Disabled: hit was used to end episode when bumping into an object
        self.hit = "none"
        # Cross track error
        self.cte = 0.0

        self.materials_queue_size = 10
        self.max_frames_on_ground = 30
        if MAX_THROTTLE < 0.5 and self.max_frames_on_ground == 30:
            self.logger.warn(
                "Change max_frames_on_ground since it was set with MAX_THROTTLE = 0.5"
            )

        self.initialize_materials()
        self.index_materials = 0

        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0

        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0
        self.rot_w = 0.0

        self.total_nodes = 0
        self.current_track_string = "none"
        self.configuration: EnvConfiguration = None

        self.reward = 0.0
        self.prev_reward = 0.0
        self.waypoints_crossed = []
        self.start_time = time.time()

        # AVF logs
        self.car_trajectory: List[Tuple[float, float]] = []
        self.images: List[str] = []
        self.rewards: List[float] = []
        self.actions: List[Tuple[float, float]] = []
        self.agent_state: dict = dict()
        self.reconstruction_losses: List[float] = []
        self.steerings: List[float] = []
        self.speeds: List[float] = []
        self.ctes: List[float] = []
        self.fitness_values: List[float] = []

        self.steering_angle = 0.0
        self.current_step = 0
        self.total_steps = 0
        self.speed = 0
        self.steering = 0.0
        self.last_steering = 0.0

        self.frame_queue = Queue(1)
        self.track_strings = []
        self.start = True
        self.frame_count = 0

        self.is_paused = False
        # Define which method should be called
        # for each type of message
        self.fns = {
            "telemetry": self.on_telemetry,
            "scene_selection_ready": self.on_scene_selection_ready,
            "scene_names": self.on_recv_scene_names,
            "car_loaded": self.on_car_loaded,
            "send_track": self.on_recv_track,
            "need_car_config": self.on_need_car_config,
        }

    def initialize_materials(self):
        self.materials = ["none"] * self.materials_queue_size

    def append_to_materials(self, material: str):
        if self.index_materials == self.materials_queue_size:
            self.index_materials = 0
        self.materials[self.index_materials] = material
        self.index_materials += 1

    def check_ground(self):
        for material in self.materials:
            if material == "ground":
                return True
        return False

    def check_distance_from_center(self):
        return abs((self.cte / self.max_cte_error) + 0.2) > 0.6

    def check_line(self):
        for material in self.materials:
            if material == "end_line":
                return True
        return False

    def send_cam_config(
        self,
        img_w=0,
        img_h=0,
        img_d=0,
        img_enc=0,
        fov=0,
        fish_eye_x=0,
        fish_eye_y=0,
        offset_x=0,
        offset_y=0,
        offset_z=0,
        rot_x=0,
    ):
        """Camera config
        set any field to Zero to get the default camera setting.
        offset_x moves camera left/right
        offset_y moves camera up/down
        offset_z moves camera forward/back
        rot_x will rotate the camera
        with fish_eye_x/y == 0.0 then you get no distortion
        img_enc can be one of JPG|PNG|TGA
        """
        msg = {
            "msg_type": "cam_config",
            "fov": str(fov),
            "fish_eye_x": str(fish_eye_x),
            "fish_eye_y": str(fish_eye_y),
            "img_w": str(img_w),
            "img_h": str(img_h),
            "img_d": str(img_d),
            "img_enc": str(img_enc),
            "offset_x": str(offset_x),
            "offset_y": str(offset_y),
            "offset_z": str(offset_z),
            "rot_x": str(rot_x),
        }
        self.blocking_send(msg)
        time.sleep(0.1)

    def blocking_send(self, msg):
        if self.client is None:
            print(f"skiping: \n {msg}")
            return
        self.client.send_now(msg)

    def send_config(self):
        print("sending car config.")
        self.send_cam_config()
        print("done sending car config.")

    def on_need_car_config(self, message):
        print("on need car config")
        self.loaded = True
        self.send_config()

    def on_connect(self, client):
        """
        :param client: (client object)
        """
        self.client = client

    def on_disconnect(self):
        """
        Close client.
        """
        self.client = None

    def on_abort(self):
        self.client.stop()

    def on_recv_message(self, message):
        """
        Distribute the received message to the appropriate function.

        :param message: (dict)
        """
        if "msg_type" not in message:
            return

        msg_type = message["msg_type"]
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            print("Unknown message type", msg_type)

    def reset(self, end_of_episode: bool = False):
        """
        Global reset, notably it
        resets car to initial position.
        """
        self.send_reset_car()

        if not end_of_episode:
            self.generate_track()

        self.start = False
        self.image_array = np.zeros(self.camera_img_size)
        self.last_obs = None
        self.hit = "none"
        self.steering = 0.0
        self.last_steering = 0.0
        self.initialize_materials()
        self.index_materials = 0
        self.cte = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0

        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0
        self.rot_w = 0.0

        # AVF logs
        self.actions: List[Tuple[float, float]] = []
        self.rewards: List[float] = []
        self.images: List[str] = []
        self.car_trajectory: List[Tuple[float, float]] = []
        self.agent_state: dict = dict()
        self.is_success = 0
        self.reconstruction_losses: List[float] = []
        self.steerings: List[float] = []
        self.speeds: List[float] = []
        self.ctes: List[float] = []
        self.fitness_values: List[float] = []

        self.current_step = 0

        self.total_nodes = 0
        self.reward = 0.0
        self.prev_reward = 0.0
        self.waypoints_crossed = []

        self.control_timer.reset()
        self.observation_timer.reset()
        while not self.frame_queue.empty():
            self.frame_queue.get_nowait()
        self.frame_count = 0
        self.start = True
        self.send_restart_simulation()

    def get_sensor_size(self):
        """
        :return: (tuple)
        """
        return self.camera_img_size

    def take_action(self, action):
        """
        :param action: ([float]) Steering and throttle
        """
        throttle = action[1]
        self.steering = action[0]
        self.last_throttle = throttle
        self.actions.append((float(self.steering), float(self.last_throttle)))
        self.current_step += 1
        self.total_steps += 1
        self.send_control(self.steering, throttle)

    def observe(self):
        try:
            self.frame_queue.get(timeout=3)
        except Empty:
            self.logger.warn("Observe frame queue timeout")
            self.send_restart_simulation()

        self.last_obs = self.image_array
        observation = self.image_array
        _, loss = self.vae.get_reconstruction_and_loss(
            observation=observation, roi=False
        )
        self.reconstruction_losses.append(loss)

        done = self.is_game_over()
        reward = self.calc_reward(done)
        self.last_steering = self.steering

        if done:
            self.send_pause_simulation()
            donkey_training_logs = DonkeyTrainingLogs(
                is_success=self.is_success,
                agent_state=self.agent_state,
                config=self.configuration.str_to_config(s=self.current_track_string),
                car_trajectory=self.car_trajectory,
                images=self.images,
                fitness_values=self.fitness_values,
            )
            self.avf.store_training_logs(training_logs=donkey_training_logs)
            # FIXME: maybe observe is called multiple times in testing mode and we want to avoid training logs
            #  with no information being written (since the reset is performed before the observe method is called
            #  a second time and all the variables are re-initialized)
            if len(self.images) > 1:
                self.avf.store_testing_logs(training_logs=donkey_training_logs)
        else:
            self.rewards.append(reward)
        info = {
            "is_success": self.is_success,
            "speed": self.speed,
            "steering_angle": self.steering_angle,
        }

        if len(self.fitness_values) > 0:
            info["fitness"] = min(self.fitness_values)

        self.control_timer.on_frame()
        return observation, reward, done, info

    def is_game_over(self) -> bool:
        """
        :return: (bool)
        """
        check_distance_from_center = self.check_distance_from_center()
        check_line = self.check_line()
        if check_distance_from_center:
            self.is_success = 0
        elif check_line:
            self.is_success = 1
        return check_distance_from_center or check_line

    def calc_reward(self, done):
        """
        Compute reward:
        - +1 life bonus for each step + throttle bonus
        - -10 crash penalty - penalty for large throttle during a crash
        """

        if done:
            if self.is_success == 0:
                norm_throttle = (self.last_throttle - MIN_THROTTLE) / (
                    MAX_THROTTLE - MIN_THROTTLE
                )
                return REWARD_CRASH - CRASH_SPEED_WEIGHT * norm_throttle

        waypoint_crossed_reward = 0
        if (
            self.total_nodes > 0
            and "waypointline_" in self.hit
            and "waypointline_last_" not in self.hit
        ):
            if self.hit not in self.waypoints_crossed:
                waypoint_crossed_reward = (
                    200 * (self.simulation_mul / 4) / self.total_nodes
                    if self.simulation_mul > 4
                    else 200 / self.total_nodes
                )
                self.waypoints_crossed.append(self.hit)

        # cte_reward can be higher when the car is driving in the opposite direction; cte messes up
        cte_reward = (
            0.1
            * self.simulation_mul
            * min(abs((self.cte / self.max_cte_error) + 0.2), 0.5)
        )
        return (-0.1 * self.simulation_mul) + waypoint_crossed_reward - cte_reward

    # ------ Socket interface ----------- #

    def on_telemetry(self, data):
        """
        Update car info when receiving telemetry message.

        :param data: (dict)
        """
        img_string = data["image"]
        self.images.append(img_string)
        image = Image.open(BytesIO(base64.b64decode(img_string)))
        # Resize and crop image
        image = np.array(image)
        # Save original image for render
        self.original_image = np.copy(image)
        # Region of interest
        r = ROI
        image = image[int(r[1]) : int(r[1] + r[3]), int(r[0]) : int(r[0] + r[2])]
        # Convert RGB to BGR
        image = image[:, :, ::-1]
        self.image_array = image

        material = data["material"]
        self.append_to_materials(material=material)

        self.pos_x = data["pos_x"]
        self.pos_y = data["pos_y"]
        self.pos_z = data["pos_z"]

        self.car_trajectory.append((self.pos_x, self.pos_z))

        self.rot_x = data["rot_x"]
        self.rot_y = data["rot_y"]
        self.rot_z = data["rot_z"]
        self.rot_w = data["rot_w"]

        self.steering_angle = data["steering_angle"]
        self.steerings.append(copy.deepcopy(self.steering_angle))
        self.speed = data["speed"]
        self.speeds.append(copy.deepcopy(self.speed))

        # Cross track error not always present.
        # Will be missing if path is not setup in the given scene.
        # It should be setup in the 3 scenes available now.
        self.cte = data["cte"]
        self.ctes.append(copy.deepcopy(self.cte))

        check_distance_from_center = self.check_distance_from_center()
        if check_distance_from_center:
            self.fitness_values.append(0.0)
        else:
            fitness_value = (1 - (abs(self.cte / self.max_cte_error + 0.2) - 0.6)) - 1
            assert (
                0 <= fitness_value <= 1
            ), f"Fitness value not in bounds: {fitness_value}"
            self.fitness_values.append(fitness_value)

        self.total_nodes = data["totalNodes"]
        self.hit = data["hit"]

        if self.start and self.frame_count % 1 == 0:
            self.observation_timer.on_frame()
            self.frame_queue.put(1)
        self.frame_count += 1

    def on_scene_selection_ready(self, data):
        """
        Get the level names when the scene selection screen is ready
        """
        self.logger.info("Scene Selection Ready")
        self.send_get_scene_names()

    def on_car_loaded(self, data):
        self.loaded = True

    def on_recv_track(self, data):
        if data is not None:
            self.current_track_string = data["track_string"]

    def on_recv_scene_names(self, data):
        """
        Select the level.

        :param data: (dict)
        """
        if data is not None:
            names = data["scene_names"]
            assert (
                self.simulator_scene.get_scene_name() in names
            ), "{} not in the list of possible scenes {}".format(
                self.simulator_scene.get_scene_name(), names
            )
            self.send_load_scene(self.simulator_scene.get_scene_name())

    def generate_track(self):
        self.configuration = self.avf.generate_env_configuration()
        track_string = self.configuration.get_str()
        self.configuration.update_implementation(
            track=self.configuration.str_to_config(s=track_string).track_elements
        )
        self.send_regen_track(track_string=track_string)
        self.track_strings.append(track_string)
        max_iterations = 1000
        time_elapsed = 0
        while (
            self.track_strings[-1] != self.current_track_string and max_iterations > 0
        ):
            time.sleep(0.1)
            time_elapsed += 0.1
            if time_elapsed >= 1.0:
                time_elapsed = 0
                self.send_regen_track(track_string=track_string)
            max_iterations -= 1

        if max_iterations == 0:
            assert (
                self.track_strings[-1] == self.current_track_string
            ), "Track generated {} != {} Track deployed".format(
                self.track_strings[-1], self.current_track_string
            )

        time.sleep(1)

    def send_regen_track(self, track_string: str):
        msg = {
            "msg_type": "regen_track",
            "track_string": track_string,
            "lane_string": "right",
        }
        self.queue_message(msg)

    def send_pause_simulation(self):
        msg = {"msg_type": "pause_simulation"}
        self.queue_message(msg)
        self.is_paused = True

    def send_set_timescale(self, timescale: float):
        msg = {"msg_type": "set_timescale", "timescale": timescale.__str__()}
        self.queue_message(msg)

    def send_restart_simulation(self):
        msg = {"msg_type": "restart_simulation"}
        self.queue_message(msg)
        self.is_paused = False

    # agent info during training
    def send_agent_state(self, agent_state: Dict):
        self.agent_state = agent_state

    def send_control(self, steer, throttle, brake: float = None):
        """
        Send message to the server for controlling the car.

        :param steer: (float)
        :param throttle: (float)
        :param brake: (float)
        """
        if not self.loaded:
            return
        if brake is not None:
            msg = {
                "msg_type": "control",
                "steering": steer.__str__(),
                "throttle": throttle.__str__(),
                "brake": brake.__str__(),
            }
        else:
            msg = {
                "msg_type": "control",
                "steering": steer.__str__(),
                "throttle": throttle.__str__(),
                "brake": "0.0",
            }
        self.queue_message(msg)

    def send_reset_random_waypoint(self):
        msg = {"msg_type": "reset_random_waypoint"}
        self.queue_message(msg)

    def send_reset_car(self):
        """
        Reset car to initial position.
        """
        msg = {"msg_type": "reset_car"}
        self.queue_message(msg)

    def send_get_scene_names(self):
        """
        Get the different levels available
        """
        msg = {"msg_type": "get_scene_names"}
        self.queue_message(msg)

    def send_load_scene(self, scene_name):
        """
        Load a level.

        :param scene_name: (str)
        """
        msg = {"msg_type": "load_scene", "scene_name": scene_name}
        self.queue_message(msg)

    def send_exit_scene(self):
        """
        Go back to scene selection.
        """
        msg = {"msg_type": "exit_scene"}
        self.queue_message(msg)

    def queue_message(self, msg):
        """
        Add message to socket queue.

        :param msg: (dict)
        """
        if self.client is None:
            return

        self.client.queue_message(msg)
