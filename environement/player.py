class Player(object):

    def __init__(self, id:int):
        self.id = id

    """
    The function evaluating if two players are equal  (for simple comparison)
    """
    def __eq__(self, other: 'Player'):
        return self.id == other.id
