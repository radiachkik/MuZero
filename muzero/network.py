# TODO: Klasse muss angepasst werdem

class Network(object):

	def initial_inference(self, image) -> NetworkOutput:
		# representation + prediction function
		return NetworkOutput(0, 0, {}, [])

	def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
		# dynamics + prediction function
		return NetworkOutput(0, 0, {}, [])

	def get_weights(self):
		# Returns the weights of this network.
		return []

	def training_steps(self) -> int:
		# How many steps / batches the network has been trained for.
		return 0