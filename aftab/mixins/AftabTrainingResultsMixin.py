from baloot import funnel
from pathlib import Path
from .AftabBaseMixin import AftabBaseMixin


class AftabTrainingResultsMixin(AftabBaseMixin):
    def __init__(self):
        super().__init__()

    def __make_log_filename(self) -> str:
        filename = f"seed-{self.buffer.seed}__"
        filename += f"environment-{self.buffer.environment}__"
        filename += f"encoder-{self.encoder.__name__}__"
        filename += f"network-{self.network}__"
        filename += f"gamma-{self.gamma}__"
        filename += f"lr-{self.lr}__"
        filename += f"epochs-{self.epochs}__"
        filename += f"lambda-{self.return_lambda}__"
        filename += f"autocast-float16-{self.autocast_float16}__"
        filename += f"channels-last-{self.channels_last}__"
        filename += f"compiled-{self.torch_compile}__"
        if self.network in ["bootstrapped", "bootstrapped-duelling"]:
            filename += f"heads-{self.bootstrap_heads}__"
            filename += f"bootstrap-p-{self.bootstrap_probability}__"

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
            "frame_skip": self.frame_skip,
            "training_reward": self.results.rewards.train,
            "test_reward": self.results.rewards.test,
            "loss": self.results.loss,
            "duration_seconds": duration,
            "duration_hours": duration / 3600,
        }

        if self.network in ["distributional", "distributional-duelling"]:
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

        if self.network in ["bootstrapped", "bootstrapped-duelling"]:
            data.update(
                {
                    "bootstrap_heads": self.bootstrap_heads,
                    "bootstrap_probability": self.bootstrap_probability,
                }
            )

        return data

    def __create_directory(self, directory_path: str) -> str:
        directory_path = directory_path.replace(".", "/")
        Path(directory_path).mkdir(exist_ok=True, parents=True)
        return directory_path

    def _log(self, *, directory: str) -> None:
        directory_path = self.__create_directory(directory).strip("/").strip()
        filename = self.__make_log_filename()
        payload = self.__build_log_payload()
        funnel(f"{directory_path}/{filename}", payload)
