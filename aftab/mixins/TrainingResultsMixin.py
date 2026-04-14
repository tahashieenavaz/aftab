from baloot import funnel


class TrainingResultsMixin:
    def __init__(self):
        super().__init__()

    def flush_final_properties(self):
        self.final_training_rewards = None
        self.final_test_rewards = None
        self.final_loss_evolution = None
        self.final_duration = None

    def make_log_filename(self, **kwargs):
        filename = "_".join(f"{k}-{v}" for k, v in kwargs.items())
        return f"{filename}.pkl"

    def _build_log_payload(self):
        duration = self.final_duration or 0
        return {
            "training_reward": self.final_training_rewards,
            "test_reward": self.final_test_rewards,
            "loss": self.final_loss_evolution,
            "duration_seconds": duration,
            "duration_hours": duration / 3600,
        }

    def save(self, **kwargs) -> None:
        filename = self.make_log_filename(**kwargs)
        payload = self._build_log_payload()
        funnel(filename, payload)
