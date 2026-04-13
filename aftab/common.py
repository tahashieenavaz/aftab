class LinearEpsilon:
    def __init__(self, ratio: float = 0.1, target=0.001, maximum: float = 1.0):
        self.maximum = maximum
        self.target = target
        self.ratio = ratio

    def get(self, frames, total_frames, all_rewards, episode_returns):
        target = self.target
        top = self.top
        decay_duration = total_frames * self.ratio
        if decay_duration == 0:
            return top
        return max(target, top - (frames / decay_duration) * (top - target))
