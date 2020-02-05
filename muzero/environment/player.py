class Player(object):
    """
    A class representing the available players in the current game
    """

    def __init__(self, player_id: int):
        """
        :param player_id: The integer value associated with this player
        """
        self.player_id = player_id

    def __eq__(self, other: 'Player'):
        """
        :param other: The other player to compare this one with
        :return: A bool indicating whether this player is equal to the other one
        """
        return self.player_id == other.player_id
