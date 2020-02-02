from network import Network

"""
Storage for all saved versions of the network
"""
class NetworkStorage(object):

	def __init__(self):
		self._networks = {}

	def latest_network(self) -> Network:
		if self._networks:
			return self._networks[max(self._networks.keys())]
		else:
			# policy -> uniform, value -> 0, reward -> 0
			return make_uniform_network()

	def save_network(self, step: int, network: Network):
		self._networks[step] = network