import os

from stable_baselines3.common.callbacks import BaseCallback


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(
        self,
        log_interval: int,
        save_path: str,
        num_envs: int = 1,
        name_prefix=None,
        verbose=0,
    ):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.log_interval = log_interval // num_envs
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            if self.name_prefix is not None:
                path = os.path.join(
                    self.save_path,
                    "{}_{}_steps.pkl".format(self.name_prefix, self.num_timesteps),
                )
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print("Saving VecNormalize to {}".format(path))
        return True
