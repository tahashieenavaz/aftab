from .InvokableMixin import InvokableMixin


class CheckFramesMixin(InvokableMixin):
    acceptable_frames_idx = {
        "pilot": 50_000_000,
        "ablation": 50_000_000,
        "full": 200_000_000,
    }

    def __init__(self):
        super().__init__()

        if not isinstance(self.frames, str):
            return

        if self.frames not in CheckFramesMixin.acceptable_frames_idx:
            raise ValueError(
                f"Total frames was passed a wrong value of `{self.frames}`. Acceptable values are `pilot`, `ablation`, `full`."
            )

        fetched_frames = CheckFramesMixin.acceptable_frames_idx.get(self.frames)
        self.frames = fetched_frames
