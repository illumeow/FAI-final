class MinPlayer:
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def action(self, hand, history):
        # Hands are sorted by the engine.
        return hand[0]
