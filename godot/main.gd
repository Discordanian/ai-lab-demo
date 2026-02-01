extends Control

var nn: NeuralNetwork
@onready var matrix1: GridContainer = $MarginContainer/VBoxContainer/Weights/VBoxContainer/HBoxContainer/Matrix1
@onready var matrix2: GridContainer = $MarginContainer/VBoxContainer/Weights/VBoxContainer/HBoxContainer/Matrix2
@onready var randomize_button: Button = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/VBoxContainer/Randomize
@onready var validate_button: Button = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/VBoxContainer/Validate
@onready var step_train_button: Button = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/VBoxContainer/StepTrain
@onready var auto_train_button: Button = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/VBoxContainer/AutoTrain
@onready var output_false_false: Label = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/Results/FalseFalse/Output
@onready var output_false_true: Label = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/Results/FalseTrue/Output
@onready var output_true_false: Label = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/Results/TrueFalse/Output
@onready var output_true_true: Label = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/Results/TrueTrue/Output
@onready var expected_false_false: Label = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/Results/FalseFalse/Expected
@onready var expected_false_true: Label = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/Results/FalseTrue/Expected
@onready var expected_true_false: Label = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/Results/TrueFalse/Expected
@onready var expected_true_true: Label = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/Results/TrueTrue/Expected
@onready var color_rect_headers: ColorRect = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/Results/Headers/ColorRect
@onready var color_rect_false_false: ColorRect = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/Results/FalseFalse/ColorRect
@onready var color_rect_false_true: ColorRect = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/Results/FalseTrue/ColorRect
@onready var color_rect_true_false: ColorRect = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/Results/TrueFalse/ColorRect
@onready var color_rect_true_true: ColorRect = $MarginContainer/VBoxContainer/Controls_Results/HBoxContainer/Results/TrueTrue/ColorRect
@onready var total_runs_label: Label = $MarginContainer/VBoxContainer/MarginContainer/HBoxContainer/TotalRuns

const COLOR_GREEN := Color(0.2, 0.8, 0.2, 1.0)
const COLOR_RED := Color(0.9, 0.2, 0.2, 1.0)

@export var epochs: int = 4
@export var tick_interval: float = 0.3

var _auto_train_timer: Timer


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	nn = NeuralNetwork.new(2, 2, 1)
	_update_weight_labels()
	_update_total_runs_label()
	randomize_button.pressed.connect(_on_randomize_weights_pressed)
	validate_button.pressed.connect(_on_validate_pressed)
	step_train_button.pressed.connect(_on_step_train_pressed)
	auto_train_button.pressed.connect(_on_auto_train_pressed)

	_auto_train_timer = Timer.new()
	_auto_train_timer.one_shot = false
	add_child(_auto_train_timer)
	_auto_train_timer.timeout.connect(_on_auto_train_tick)
	_on_validate_pressed()

func _on_randomize_weights_pressed() -> void:
	nn = NeuralNetwork.new(2, 2, 1)
	_update_weight_labels()
	_update_total_runs_label()


func _on_step_train_pressed() -> void:
	var inputs: Array = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
	var targets: Array = [[0.0], [1.0], [1.0], [0.0]]
	for e in range(epochs):
		for i in range(4):
			nn.train(inputs, targets)
	_update_weight_labels()
	_update_total_runs_label()
	_on_validate_pressed()


func _on_validate_pressed() -> void:
	var result_00: Array = nn.forward([[0.0, 0.0]])
	output_false_false.text = "TRUE" if result_00[0][0] >= 0.5 else "FALSE"
	var result_01: Array = nn.forward([[0.0, 1.0]])
	output_false_true.text = "TRUE" if result_01[0][0] >= 0.5 else "FALSE"
	var result_10: Array = nn.forward([[1.0, 0.0]])
	output_true_false.text = "TRUE" if result_10[0][0] >= 0.5 else "FALSE"
	var result_11: Array = nn.forward([[1.0, 1.0]])
	output_true_true.text = "TRUE" if result_11[0][0] >= 0.5 else "FALSE"

	var match_00: bool = output_false_false.text == expected_false_false.text
	var match_01: bool = output_false_true.text == expected_false_true.text
	var match_10: bool = output_true_false.text == expected_true_false.text
	var match_11: bool = output_true_true.text == expected_true_true.text

	color_rect_false_false.color = COLOR_GREEN if match_00 else COLOR_RED
	color_rect_false_true.color = COLOR_GREEN if match_01 else COLOR_RED
	color_rect_true_false.color = COLOR_GREEN if match_10 else COLOR_RED
	color_rect_true_true.color = COLOR_GREEN if match_11 else COLOR_RED

	var all_match: bool = match_00 and match_01 and match_10 and match_11
	color_rect_headers.color = COLOR_GREEN if all_match else COLOR_RED


func _update_total_runs_label() -> void:
	total_runs_label.text = str(nn.get_training_runs())


func _on_auto_train_pressed() -> void:
	_auto_train_timer.wait_time = tick_interval
	_auto_train_timer.start()


func _on_auto_train_tick() -> void:
	_on_step_train_pressed()
	_on_validate_pressed()
	var all_pass: bool = (
		output_false_false.text == expected_false_false.text
		and output_false_true.text == expected_false_true.text
		and output_true_false.text == expected_true_false.text
		and output_true_true.text == expected_true_true.text
	)
	if all_pass:
		_auto_train_timer.stop()
		print("Total training runs: ", nn.get_training_runs())


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
