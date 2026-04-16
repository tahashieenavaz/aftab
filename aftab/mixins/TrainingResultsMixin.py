from baloot import funnel
from types import SimpleNamespace


class TrainingResultsMixin:
    def __init__(self):
        super().__init__()

    def flush_results(self):
        self.results = SimpleNamespace()
        self.results.rewards = SimpleNamespace()
        self.results.rewards.train = None
        self.results.rewards.test = None
        self.results.loss = None
        self.results.duration = None

    def make_log_filename(self, **kwargs):
        filename = "_".join(f"{k}-{v}" for k, v in kwargs.items())
        return f"{filename}.pkl"

    def _build_log_payload(self):
        duration = self.results.duration or 0
        return {
            "training_reward": self.results.rewards.train,
            "test_reward": self.results.rewards.test,
            "loss": self.results.loss,
            "duration_seconds": duration,
            "duration_hours": duration / 3600,
        }

    def save(self, **kwargs) -> None:
        filename = self.make_log_filename(**kwargs)
        payload = self._build_log_payload()
        funnel(filename, payload)
