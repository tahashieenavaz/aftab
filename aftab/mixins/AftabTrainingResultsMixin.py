from baloot import funnel
from aftab.common import _make_sure_directory_exists
from .AftabBaseMixin import AftabBaseMixin


class AftabTrainingResultsMixin(AftabBaseMixin):
    def __init__(self):
        super().__init__()

    def __make_log_filename(self) -> str:
        filename = f"experiment-{self.experiment_name}__"
        filename += f"network-{self.network}__"
        filename += f"seed-{self.buffer.seed}__"
        filename += f"environment-{self.buffer.environment}__"
        filename += f"encoder-{self.encoder.__name__}__"
        # removes trailing __
        filename = filename.strip("__")
        return f"{filename}.pkl"

    def __build_log_payload(self) -> dict:
        duration = self.results.duration or 0
        data = {
            "environment": self.buffer.environment,
            "seed": self.buffer.seed,
            "encoder": self.encoder.__name__,
            "optimizer": self.optimizer,
            "lambda": self.return_lambda,
            "network": self.network,
            "frames": self.frames,
            "frame_skip": self.environment_frame_skip,
            "training_reward": self.results.rewards.train,
            "test_reward": self.results.rewards.test,
            "loss": self.results.loss,
            "duration_seconds": duration,
            "duration_hours": duration / 3600,
        }

        if self.network in [
            "distributional",
            "distributional-duelling",
            "distributional-bootstrapped-duelling",
        ]:
            data.update(
                {
                    "distributional_bins": self.distributional_bins,
                    "distributional_min_value": self.distributional_min_value,
                    "distributional_max_value": self.distributional_max_value,
                    "distributional_sigma": self.distributional_sigma,
                    "distributional_sigma_ratio": self.distributional_sigma_ratio,
                    "distributional_value_clip": self.distributional_value_clip,
                }
            )

        if self.network in [
            "bootstrapped",
            "bootstrapped-duelling",
            "distributional-bootstrapped-duelling",
        ]:
            data.update(
                {
                    "bootstrap_heads": self.bootstrap_heads,
                    "bootstrap_probability": self.bootstrap_probability,
                }
            )

        return data

    def _log(self, *, directory: str) -> None:
        directory_path = _make_sure_directory_exists(directory).strip("/").strip()
        filename = self.__make_log_filename()
        payload = self.__build_log_payload()
        funnel(f"{directory_path}/{filename}", payload)
