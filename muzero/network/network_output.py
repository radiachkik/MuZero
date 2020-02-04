from muzero.environment.action import Action
from typing import Dict, List, NamedTuple


class NetworkOutput(NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]