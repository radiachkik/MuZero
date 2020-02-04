class Action(object):

    def __init__(self, action_id: int):
        self.action_id = action_id

    """
    Returning the int representation of an action used to compare dictionary keys during a dictionary lookup
    """
    def __hash__(self):
        return self.action_id

    """
    The function evaluating if two actions are equal  (for simple comparison)
    """
    def __eq__(self, other: 'Action'):
        return self.action_id == other.action_id
