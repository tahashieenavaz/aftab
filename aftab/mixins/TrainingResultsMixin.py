from baloot import funnel
from pathlib import Path


class TrainingResultsMixin:
    def __init__(self):
        super().__init__()

    def __make_log_filename(self) -> str:
        filename = f"seed-{self.buffer.seed}__"
        filename += f"environment-{self.buffer.environment}__"
        filename += f"encoder-{self.encoder.__name__}__"
        filename += f"network-{self.network}__"
        filename += f"gamma-{self.gamma}__"
        filename += f"lr-{self.lr}__"

        if self.network in ["fqf", "dfqf"]:
            filename += f"quantiles-{self.number_quantiles}__"
            filename += f"quantile-dimension-{self.quantile_embedding_dimension}__"
            filename += f"return-steps-{self.steps}__"
        else:
            filename += f"lambda-{self.lmbda}__"

        if self.reward_centering:
            filename += f"reward-centering-beta-{self.reward_centering}__"

        if self.random_shift:
            filename += f"random-shift-padding_{self.random_shift_padding}__"
            filename += f"random-shift-m-{self.random_shift_m}__"
            filename += f"random-shift-k-{self.random_shift_k}__"

        # removes trailing __
        filename = filename.strip("__")

        return f"{filename}.pkl"

    def __build_log_payload(self) -> dict:
        duration = self.results.duration or 0
        return {
            "training_reward": self.results.rewards.train,
            "test_reward": self.results.rewards.test,
            "loss": self.results.loss,
            "duration_seconds": duration,
            "duration_hours": duration / 3600,
        }

    def __create_directory(self, directory_path: str) -> str:
        directory_path = directory_path.replace(".", "/")
        Path(directory_path).mkdir(exist_ok=True, parents=True)
        return directory_path

    def log(self, directory: str = "results") -> None:
        directory_path = self.__create_directory(directory).strip("/").strip()
        filename = self.__make_log_filename()
        payload = self.__build_log_payload()
        funnel(f"{directory_path}/{filename}", payload)
