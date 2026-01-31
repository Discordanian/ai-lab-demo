extends Control

var nn: NeuralNetwork
@onready var matrix1: GridContainer = $MarginContainer/VBoxContainer/Weights/VBoxContainer/HBoxContainer/Matrix1


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	nn = NeuralNetwork.new(2, 2, 1)
	_update_weight_labels()


func _update_weight_labels() -> void:
	var input_hidden: Array = nn.get_weights()["input_hidden"]
	print("Input hidden: ", input_hidden)
	var idx: int = 0
	for i in input_hidden.size():
		for j in input_hidden[i].size():
			matrix1.get_child(idx).text = "%.4f" % input_hidden[i][j]
			idx += 1


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
