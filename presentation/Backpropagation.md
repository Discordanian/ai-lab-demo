# Backpropagation

Given the result from the [Loss Calculation](Loss_Calculation.md) function we work backwards
through the network.  The weights are shifted proportional to the contribution to the
error.  So inputs that contributed the most error to a given node have weights changed 
the most.  Inputs with small contributions to error are changed least.  

We continue the process of updating weights going backward through the network.

# Gradient Descent
This is a deep dive topic all on its own.  

If we consider the Loss Calculation we can consider it a vector with a magnitude equal to the square of the error (ie: a postive value).  The _direction_ of the vector in Gradient Descent always points in the direction of the *greatest* change.  You can think of this as multidimensional topographical map and the vector will point in the direction of the steepest path down towards a zero error.

This allows all weights to be adjusted in the direction that most quickly reduces the loss.  The size of the update is proportional to how senstive the loss is to that weight.
