class Action(object):
    """
    A class representing one of the possible actions in the environment
    """

    def __init__(self, action_id: int):
        """
        :param action_id: the integer value associated with this action (only 'one-value' actions are supported)
        """
        self.action_id = action_id

    def __hash__(self) -> int:
        """
        :return: the integer value associated with this action
        """
        return self.action_id

    def __eq__(self, other: 'Action') -> bool:
        """
        :param other: The other action to compare this one with
        :return: A bool indicating whether this action is equal to the compared one
        """
        return self.action_id == other.action_id
