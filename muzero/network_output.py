# TODO: Klasse muss angepasst werdem

import typing
from action import Action

class NetworkOutput(typing.NamedTuple):
  value: float
  reward: float
  policy_logits: Dict[Action, float]
  hidden_state: List[float]