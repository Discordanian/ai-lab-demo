extends Node

## Autoload singleton providing global access to the neural network.
## Call create_network() to initialize, then access via .network

var network: NeuralNetwork = null

func create_network(input_size: int, hidden_size: int, output_size: int) -> NeuralNetwork:
	network = NeuralNetwork.new(input_size, hidden_size, output_size)
	return network
