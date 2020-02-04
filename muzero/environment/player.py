class Player(object):

    def __init__(self, player_id: int):
        self.player_id = player_id

    """
    The function evaluating if two players are equal  (for simple comparison)
    """
    def __eq__(self, other: 'Player'):
        return self.player_id == other.player_id
