class_name NeuralNetwork
extends RefCounted

# Weight matrices: input_size x hidden_size, hidden_size x output_size
var weights_input_hidden: Array = []  # Array of Array (2D)
var weights_hidden_output: Array = []
# Bias vectors: one per neuron in hidden layer and output layer
var bias_hidden: Array = []   # 1D, length hidden_size
var bias_output: Array = []   # 1D, length output_size
var training_runs: int = 0


func _init(input_size: int, hidden_size: int, output_size: int) -> void:
	# weights_input_hidden: [input_size][hidden_size]
	weights_input_hidden = _rand_matrix(input_size, hidden_size)
	# weights_hidden_output: [hidden_size][output_size]
	weights_hidden_output = _rand_matrix(hidden_size, output_size)
	bias_hidden = _zero_vector(hidden_size)
	bias_output = _zero_vector(output_size)
	training_runs = 0


# Maps floating point values to a value between 0 and 1.
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
			m[i].append(randf() - 0.5)
	return m


static func _rand_vector(size: int) -> Array:
	var v: Array = []
	for i in size:
		v.append(randf() - 0.5)
	return v


static func _zero_vector(size: int) -> Array:
	var v: Array = []
	for i in size:
		v.append(0.0)
	return v


static func _add_bias_to_rows(m: Array, bias: Array) -> Array:
	# m: [rows][cols], bias: [cols]. Add bias to each row.
	var out: Array = []
	for i in m.size():
		out.append([])
		for j in m[i].size():
			out[i].append(m[i][j] + bias[j])
	return out


static func _sum_columns(m: Array) -> Array:
	# Sum each column -> 1D array of length cols (for bias gradient).
	var cols: int = m[0].size()
	var out: Array = []
	for j in cols:
		var s: float = 0.0
		for i in m.size():
			s += m[i][j]
		out.append(s)
	return out


static func _add_vector(a: Array, b: Array) -> Array:
	var out: Array = []
	for i in a.size():
		out.append(a[i] + b[i])
	return out


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



## Forward pass: inputs 2D array [samples][input_size], returns [samples][output_size].
func forward(inputs: Array) -> Array:
	var hidden_input: Array = _matmul(inputs, weights_input_hidden)
	hidden_input = _add_bias_to_rows(hidden_input, bias_hidden)
	var hidden_output: Array = _sigmoid_matrix(hidden_input)
	var output_input: Array = _matmul(hidden_output, weights_hidden_output)
	output_input = _add_bias_to_rows(output_input, bias_output)
	var output: Array = _sigmoid_matrix(output_input)
	return output


## Returns all current weights as a Dictionary: "input_hidden", "hidden_output" (2D), "bias_hidden", "bias_output" (1D).
func get_weights() -> Dictionary:
	return {
		"input_hidden": weights_input_hidden,
		"hidden_output": weights_hidden_output,
		"bias_hidden": bias_hidden,
		"bias_output": bias_output
	}

func get_training_runs() -> int:
	return training_runs

func reset_training_runs() -> void:
	training_runs = 0

## Train via backpropagation for a single pass. inputs/targets are 2D arrays.
func train(inputs: Array, targets: Array) -> void:
	training_runs += 1
	var hidden_input: Array = _matmul(inputs, weights_input_hidden)
	hidden_input = _add_bias_to_rows(hidden_input, bias_hidden)
	var hidden_output: Array = _sigmoid_matrix(hidden_input)
	var output_input: Array = _matmul(hidden_output, weights_hidden_output)
	output_input = _add_bias_to_rows(output_input, bias_output)
	var output: Array = _sigmoid_matrix(output_input)

	var error: Array = _sub_matrix(targets, output)
	var d_output: Array = _mul_elementwise(error, _sigmoid_derivative_matrix(output))

	var error_hidden: Array = _matmul(d_output, _transpose(weights_hidden_output))
	var d_hidden: Array = _mul_elementwise(error_hidden, _sigmoid_derivative_matrix(hidden_output))

	var delta_ho: Array = _matmul(_transpose(hidden_output), d_output)
	var delta_ih: Array = _matmul(_transpose(inputs), d_hidden)
	var delta_bias_output: Array = _sum_columns(d_output)
	var delta_bias_hidden: Array = _sum_columns(d_hidden)

	weights_hidden_output = _add_matrix(weights_hidden_output, delta_ho)
	weights_input_hidden = _add_matrix(weights_input_hidden, delta_ih)
	bias_output = _add_vector(bias_output, delta_bias_output)
	bias_hidden = _add_vector(bias_hidden, delta_bias_hidden)
