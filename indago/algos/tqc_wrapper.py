"""
The MIT License

Copyright (c) 2019 Antonin Raffin

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

import io
import pathlib
from typing import Any, Dict, Optional, Union

import numpy as np
import torch as th
from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.save_util import recursive_setattr
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import check_for_correct_spaces, polyak_update

from indago.algos.tqc import TQC
from indago.algos.tqc_utils import quantile_huber_loss
from indago.env_wrapper import EnvWrapper
from indago.utils.dummy_c_vec_env import DummyCVecEnv
from indago.utils.save_util import load_from_zip_file


class TQCWrapper(TQC):
    def __init__(self, **kwargs):
        super(TQCWrapper, self).__init__(**kwargs)

    # FIXME: did not find a better solution: what I need is a custom version of load_from_zip_file function
    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> BaseAlgorithm:
        """
        Load the model from a zip-file

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        data, params, pytorch_variables = load_from_zip_file(
            path, device=device, custom_objects=custom_objects, algo_name="tqc"
        )

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # noinspection PyArgumentList
        model = cls(  # pytype: disable=not-instantiable,wrong-keyword-args
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        # put state_dicts back in place
        model.set_parameters(params, exact_match=True, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                recursive_setattr(model, name, pytorch_variables[name])

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # pytype: disable=attribute-error
        return model

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())
            self.replay_buffer.ent_coef = ent_coef.item()

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                stddev = self.actor.action_dist.distribution.stddev
                stddev = stddev.cpu().numpy().squeeze().tolist()

                # Compute and cut quantiles at the next state
                # batch x nets x quantiles
                next_quantiles = self.critic_target(replay_data.next_observations, next_actions)

                # Sort and drop top k quantiles to control overestimation.
                n_target_quantiles = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
                next_quantiles, _ = th.sort(next_quantiles.reshape(batch_size, -1))
                next_quantiles = next_quantiles[:, :n_target_quantiles]

                # td error + entropy term
                target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_quantiles
                # Make target_quantiles broadcastable to (batch_size, n_critics, n_target_quantiles).
                target_quantiles.unsqueeze_(dim=1)

            # Get current Quantile estimates using action from the replay buffer
            current_quantiles = self.critic(replay_data.observations, replay_data.actions)
            # Compute critic loss, not summing over the quantile dimension as in the paper.
            critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            qf_pi = self.critic(replay_data.observations, actions_pi).mean(dim=2).mean(dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        agent_state = dict()
        agent_state["training_progress"] = (self.num_timesteps / self._total_timesteps) * 100
        agent_state["ent_coef"] = ent_coef.item()

        env_unwrapped = self.get_env().unwrapped
        while not isinstance(env_unwrapped, DummyCVecEnv):
            env_unwrapped = env_unwrapped.unwrapped

        assert isinstance(env_unwrapped.envs[0].env, EnvWrapper), "{} is not an instance of EnvWrapper".format(
            type(env_unwrapped.envs[0].env)
        )
        env_unwrapped.envs[0].env.send_agent_state(agent_state=agent_state)

        # self.env.unwrapped.send_agent_state(agent_state=agent_state)

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/ent_coef", np.mean(ent_coefs))
        logger.record("train/actor_loss", np.mean(actor_losses))
        logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
