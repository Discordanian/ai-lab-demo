# Training Loop

## Forward
Using known inputs and expected outputs feed the network [Forward](Forward.md).


## Loss
Calculate the difference between the expected output and the predicted output.  [Loss](Loss_Calculation.md) returns a vector of error.

## Backpropagation
[Backpropagation](Backpropagation.md) adjusts weights and bias using Gradient Descent to reduce
the loss function.

## Epochs
Often times for training the same input is fed through a number of times for each training session.
It is not uncommon for training epochs to be in the thousands.

## Training Strategy

### Conventional
Traditionally the input domain is large and unknown.  This neural network is designed for
classification.  The classificatin is from a finite set of options but the input is often 
over a large domain where we do not have expected outcomes for each input.


### XOR example
We have the complete domain available to us.  There are only 4 different inputs possible so each 
loop through we train on all 4.  
