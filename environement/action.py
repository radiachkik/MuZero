class Action(object):

    def __init__(self, index: int):
        self.index = index

    """
    Returning the int representation of an action used to compare dictionary keys during a dictionary lookup
    """
    def __hash__(self):
        return self.index

    """
    The function evaluating if two actions are equal  (for simple comparison)
    """
    def __eq__(self, other: 'Action'):
        return self.index == other.index

    """
    The function evaluating if this action is greater then another (for simple comparison)
    """
    def __gt__(self, other: 'Action'):
        return self.index > other.index
