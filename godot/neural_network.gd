class_name NeuralNetwork
extends RefCounted

## A feedforward neural network with one hidden layer, matching nn.py behavior.
## Uses sigmoid activation and trains via backpropagation (gradient descent).

# Weight matrices: input_size x hidden_size, hidden_size x output_size
var weights_input_hidden: Array = []  # Array of Array (2D)
var weights_hidden_output: Array = []


func _init(input_size: int, hidden_size: int, output_size: int) -> void:
	# weights_input_hidden: [input_size][hidden_size]
	weights_input_hidden = _rand_matrix(input_size, hidden_size)
	# weights_hidden_output: [hidden_size][output_size]
	weights_hidden_output = _rand_matrix(hidden_size, output_size)


# --- Activation ---
static func _sigmoid(x: float) -> float:
	return 1.0 / (1.0 + exp(-x))


static func _sigmoid_derivative(x: float) -> float:
	return x * (1.0 - x)


# --- Matrix helpers ---
static func _rand_matrix(rows: int, cols: int) -> Array:
	var m: Array = []
	for i in rows:
		m.append([])
		for j in cols:
			m[i].append(randf())
	return m


static func _matmul(a: Array, b: Array) -> Array:
	# a: [rows_a][cols_a], b: [cols_a][cols_b] -> [rows_a][cols_b]
	var rows_a: int = a.size()
	var cols_a: int = a[0].size()
	var cols_b: int = b[0].size()
	var out: Array = []
	for i in rows_a:
		out.append([])
		for j in cols_b:
			var s: float = 0.0
			for k in cols_a:
				s += a[i][k] * b[k][j]
			out[i].append(s)
	return out


static func _transpose(m: Array) -> Array:
	var rows: int = m.size()
	var cols: int = m[0].size()
	var out: Array = []
	for j in cols:
		out.append([])
		for i in rows:
			out[j].append(m[i][j])
	return out


static func _sigmoid_matrix(m: Array) -> Array:
	var out: Array = []
	for i in m.size():
		out.append([])
		for j in m[i].size():
			out[i].append(_sigmoid(m[i][j]))
	return out


static func _sigmoid_derivative_matrix(m: Array) -> Array:
	var out: Array = []
	for i in m.size():
		out.append([])
		for j in m[i].size():
			out[i].append(_sigmoid_derivative(m[i][j]))
	return out


static func _mul_elementwise(a: Array, b: Array) -> Array:
	var out: Array = []
	for i in a.size():
		out.append([])
		for j in a[i].size():
			out[i].append(a[i][j] * b[i][j])
	return out


static func _sub_matrix(a: Array, b: Array) -> Array:
	var out: Array = []
	for i in a.size():
		out.append([])
		for j in a[i].size():
			out[i].append(a[i][j] - b[i][j])
	return out


static func _add_matrix(a: Array, b: Array) -> Array:
	var out: Array = []
	for i in a.size():
		out.append([])
		for j in a[i].size():
			out[i].append(a[i][j] + b[i][j])
	return out


# --- Public API ---

## Forward pass: inputs 2D array [samples][input_size], returns [samples][output_size].
func forward(inputs: Array) -> Array:
	var hidden_input: Array = _matmul(inputs, weights_input_hidden)
	var hidden_output: Array = _sigmoid_matrix(hidden_input)
	var output_input: Array = _matmul(hidden_output, weights_hidden_output)
	var output: Array = _sigmoid_matrix(output_input)
	return output


## Returns all current weights as a Dictionary: "input_hidden" (2D Array), "hidden_output" (2D Array).
func get_weights() -> Dictionary:
	return {
		"input_hidden": weights_input_hidden,
		"hidden_output": weights_hidden_output
	}


## Train via backpropagation for a single pass. inputs/targets are 2D arrays.
func train(inputs: Array, targets: Array) -> void:
	var hidden_input: Array = _matmul(inputs, weights_input_hidden)
	var hidden_output: Array = _sigmoid_matrix(hidden_input)
	var output_input: Array = _matmul(hidden_output, weights_hidden_output)
	var output: Array = _sigmoid_matrix(output_input)

	var error: Array = _sub_matrix(targets, output)
	var d_output: Array = _mul_elementwise(error, _sigmoid_derivative_matrix(output))

	var error_hidden: Array = _matmul(d_output, _transpose(weights_hidden_output))
	var d_hidden: Array = _mul_elementwise(error_hidden, _sigmoid_derivative_matrix(hidden_output))

	var delta_ho: Array = _matmul(_transpose(hidden_output), d_output)
	var delta_ih: Array = _matmul(_transpose(inputs), d_hidden)

	weights_hidden_output = _add_matrix(weights_hidden_output, delta_ho)
	weights_input_hidden = _add_matrix(weights_input_hidden, delta_ih)
