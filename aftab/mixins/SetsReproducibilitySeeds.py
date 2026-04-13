from baloot import seed_everything


class SetsReproducibilitySeeds:
    def set_seed(self, seed: int):
        seed_everything(seed)
