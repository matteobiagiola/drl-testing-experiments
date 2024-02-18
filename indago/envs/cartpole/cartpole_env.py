"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R

The MIT License

Copyright (c) 2016 OpenAI
Copyright (c) 2022 Farama Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(
        self,
        masscart: float = 1.0,
        masspole: float = 0.1,
        length: float = 0.5,
        discrete_action_space: bool = True,
        manual: bool = False,
        cart_friction: float = 0.0,
        x: float = None,
        x_dot: float = None,
        theta: float = None,
        theta_dot: float = None,
    ):
        self.gravity = 9.8
        self.masscart = masscart
        self.masspole = masspole
        self.length = length  # actually half the pole's length
        self.cart_friction = cart_friction
        self.pole_angle = 12
        # Angle at which to fail the episode
        self.theta_threshold_radians = self.pole_angle * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.discrete_action_space = discrete_action_space

        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        self.manual = manual
        self.steps = 0

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ]
        )

        if discrete_action_space:
            if manual:
                self.action_space = spaces.Discrete(3)
            else:
                self.action_space = spaces.Discrete(2)
        else:
            self.action_space = spaces.Box(
                low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32
            )

        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.viewer = None
        self.x = x
        self.x_dot = x_dot
        self.theta = theta
        self.theta_dot = theta_dot
        self.state = None

        self.low = -0.05
        self.high = 0.05

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        x, x_dot, theta, theta_dot = self.state
        if self.discrete_action_space:
            if self.manual:
                if action == 0:
                    force = 0.0
                elif action == 1:
                    force = self.force_mag
                elif action == 2:
                    force = -self.force_mag
            else:
                force = self.force_mag if action == 1 else -self.force_mag
        else:
            force = self.force_mag * float(action[0])
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        # temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        temp = (
            force
            + self.polemass_length * theta_dot * theta_dot * sintheta
            - self.cart_friction * np.sign(x_dot)
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x, x_dot, theta, theta_dot)
        done = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        done = bool(done)

        self.steps += 1

        fitness_x = (
            abs((abs(x) - self.x_threshold)) / self.x_threshold
            if -self.x_threshold < x < self.x_threshold
            else 0.0
        )
        assert (
            fitness_x <= 1.0
        ), f"Fitness_x cannot be > 1.0: {fitness_x}: x {x}, threshold {self.x_threshold}"
        fitness_theta = (
            abs(abs(theta) - self.theta_threshold_radians)
            / self.theta_threshold_radians
            if -self.theta_threshold_radians < theta < self.theta_threshold_radians
            else 0.0
        )
        assert fitness_theta <= 1.0, (
            f"fitness_theta cannot be > 1.0: {fitness_theta}: theta {theta}, "
            f"threshold {self.theta_threshold_radians}"
        )
        fitness = min(fitness_x, fitness_theta)
        assert (
            fitness >= 0
        ), f"Fitness cannot be < 0: {fitness}: (1) - {fitness_x}, (2) - {fitness_theta}"

        if self.steps == 200:
            return (
                np.array(self.state),
                0.0,
                True,
                {"is_success": True, "fitness": fitness},
            )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. "
                    "You should always call 'reset()' once you receive 'done = True' "
                    "-- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        if done:
            assert math.isclose(
                fitness, 0.0, abs_tol=0.001
            ), f"Fitness should be 0 when the pole falls. Found: {fitness}: (1) - {fitness_x}, (2) - {fitness_theta}"

        return (
            np.array(self.state),
            reward,
            done,
            {"is_success": False, "fitness": fitness},
        )

    def reset(self):
        if (
            self.x is not None
            and self.x_dot is not None
            and self.theta is not None
            and self.theta_dot is not None
        ):
            self.state = np.asarray([self.x, self.x_dot, self.theta, self.theta_dot])
        else:
            self.state = self.np_random.uniform(low=self.low, high=self.high, size=(4,))
        self.steps_beyond_done = None
        self.steps = 0
        return np.array(self.state)

    def __str__(self) -> str:
        return "(masscart: {}, masspole: {}, length: {}, cart_friction: {}, discrete_action_space: {})".format(
            self.masscart,
            self.masspole,
            self.length,
            self.cart_friction,
            self.discrete_action_space,
        )

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
