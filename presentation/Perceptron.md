# Perceptron

The base unit of Machine Learning 

It takes multipl inputs with assigned weights for each input and sums them up.  In addition to the 
inputs a weighted bias is normally added.
The sumation is sent through an [Activation Function](Sigmoid.md) 

activation_function ( x1 * w1 + x2 * w2 + x3 * w3 + . . . + bias)

![Perceptron](images/Perceptron.png)
#### Image from GeeksForGeeks

## Weights
Weights control how much each input influences the output.

## Bias
Bias controls when a perceptron activates.

# Lines
y = mx + b

m -> slope
b -> y intercept

Adding more variables with slopes looks like this:

y = m1 * x + m2 * z + b 

m1 -> x slope
m2 -> z slope
b -> y intercept

Without bias (b), the resulting line always goes through the Origin.  Think of bias as a way of shifting the line around to change the activation.
