from baloot import funnel


class SavesTrainingResults:
    def __init__(self):
        super().__init__()

    def make_log_filename(self, **arguments):
        filename = "_".join(f"{k}-{v}" for k, v in arguments.items())
        extension = "pkl"
        return f"{filename}.{extension}"

    def save(self, **arguments) -> None:
        funnel(
            self.make_log_filename(**arguments),
            {
                "training_reward": self.final_training_rewards,
                "test_reward": self.final_test_rewards,
                "loss": self.final_loss_evolution,
                "duration_seconds": self.final_duration,
                "duration_hours": self.final_duration / 3600,
            },
        )
