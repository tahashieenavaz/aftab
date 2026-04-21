from baloot import funnel
from types import SimpleNamespace


class TrainingResultsMixin:
    def __init__(self):
        super().__init__()

    def __make_log_filename(self):
        filename = f"seed-{self.buffer.seed}_"
        filename = f"network-{self.network}_"
        filename = f"environment-{self.buffer.environment}_"
        filename = f"network-{self.network}"
        return f"{filename}.pkl"

    def __build_log_payload(self):
        duration = self.results.duration or 0
        return {
            "training_reward": self.results.rewards.train,
            "test_reward": self.results.rewards.test,
            "loss": self.results.loss,
            "duration_seconds": duration,
            "duration_hours": duration / 3600,
        }

    def save(self, **kwargs) -> None:
        filename = self.__make_log_filename(**kwargs)
        payload = self.__build_log_payload()
        funnel(filename, payload)
