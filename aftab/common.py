class LinearEpsilon:
    def __init__(self, span: float = 0.1, minimum: float = 0.001, maximum: float = 1.0):
        self.maximum = maximum
        self.minimum = minimum
        self.span = span

    def get(self, frames, total_frames):
        minimum = self.minimum
        maximum = self.maximum
        decay_duration = total_frames * self.span

        if decay_duration == 0:
            return maximum

        Δ = maximum - minimum
        return max(minimum, maximum - (frames / decay_duration) * Δ)
