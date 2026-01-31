extends Control

var nn: NeuralNetwork
@onready var matrix1: GridContainer = $MarginContainer/VBoxContainer/Weights/VBoxContainer/HBoxContainer/Matrix1
@onready var matrix2: GridContainer = $MarginContainer/VBoxContainer/Weights/VBoxContainer/HBoxContainer/Matrix2
@onready var randomize_button: Button = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/VBoxContainer/Randomize
@onready var validate_button: Button = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/VBoxContainer/Validate
@onready var output_false_false: Label = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/Results/FalseFalse/Output


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	nn = NeuralNetwork.new(2, 2, 1)
	_update_weight_labels()
	randomize_button.pressed.connect(_on_randomize_weights_pressed)
	validate_button.pressed.connect(_on_validate_pressed)


func _on_randomize_weights_pressed() -> void:
	nn = NeuralNetwork.new(2, 2, 1)
	_update_weight_labels()


func _on_validate_pressed() -> void:
	var result: Array = nn.forward([[0.0, 0.0]])
	var value: float = result[0][0]
	print("Result: ", result)
	output_false_false.text = "TRUE" if value >= 0.5 else "FALSE"


func _update_weight_labels() -> void:
	var weights: Dictionary = nn.get_weights()
	var input_hidden: Array = weights["input_hidden"]
	var hidden_output: Array = weights["hidden_output"]
	print("Weights input_hidden: ", input_hidden)
	print("Weights hidden_output: ", hidden_output)
	var idx: int = 0
	for i in input_hidden.size():
		for j in input_hidden[i].size():
			matrix1.get_child(idx).text = "%.4f" % input_hidden[i][j]
			idx += 1
	idx = 0
	for i in hidden_output.size():
		for j in hidden_output[i].size():
			matrix2.get_child(idx).text = "%.4f" % hidden_output[i][j]
			idx += 1


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
