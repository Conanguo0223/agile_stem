import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
)


class CurriculumLearningCallback(BaseCallback):
    """

    Args:
        look_back (int): Number of last evaluations that should be used to check if task is solved
        factor_max (float): Factor with which the threshold is multiplied to get new threshold
    """
    parent: EvalCallback

    def __init__(self, threshold, update=0.02, factor_max=np.inf, look_back=5):
        super().__init__(verbose=0)

        self.threshold = threshold
        self.look_back = look_back
        self.update = update
        self.factor = 0
        self.factor_max = factor_max

        self.rewards = []
        self.last_adjustment = 0

    def update_task(self):
        self.factor = min(self.factor + self.update, self.factor_max)
        self.parent.eval_env.env_method('increase_scrap_factor', (self.factor))
        self.parent.training_env.env_method('increase_scrap_factor', (self.factor))

    def _on_step(self) -> bool:
        self.rewards.append(self.parent.last_mean_reward)
        if self.n_calls >= self.last_adjustment + self.look_back:

            # Check if task is solved
            if np.mean(self.rewards[-self.look_back:]) >= self.threshold:
                self.update_task()
                self.last_adjustment = self.n_calls
        self.logger.record("eval/scrap_factor", self.factor)
        return True
