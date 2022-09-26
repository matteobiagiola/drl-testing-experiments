from stable_baselines3.common.callbacks import BaseCallback


class EvalBaseCallback:
    def __init__(self):
        pass

    def on_eval_start(self) -> None:
        pass

    def on_eval_episode_step(self) -> bool:
        return True


class ProgressBarCallback(BaseCallback, EvalBaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self) -> bool:
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)
        return True

    def on_eval_start(self) -> None:
        self.n_eval_episodes = 0

    def on_eval_episode_step(self) -> bool:
        self.n_eval_episodes += 1
        self._pbar.n = self.n_eval_episodes
        self._pbar.update(0)
        return True
