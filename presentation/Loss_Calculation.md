# Loss Calculation

With each prediction we see how much of an 'error' we get.  How 'far' from accurate
is the current prediction.  In training we feed [Forward](Forward.md) and compare the predicted
result to the desired result.

In this example of XOR, we have a single output, but often times we have many outputs and the Loss
calculation is the sum of all the differences.  To avoid errors canceling each other out, the
difference is squared to force it to a positve number.  Squaring also emphasizes larger errors.



