# Forward

Probably the easiest step after initializing the network.  We feed in input values and 
see what the output is.

The weights and basis are randomized.  The output node, thanks to the [Sigmoid](Sigmoid.md) activation function, will return a value between 0 and 1.  In the case of XOR, we want to have only two values, True or False.  We'll consider an output of > 0.5 to be 'TRUE' otherwise it'll be a prediction of 'FALSE'.



