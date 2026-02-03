# MLP vs LLM

# Architecture
MLP - Feed forward network where data moves in one direction.  From input to output/prediction.  Each node is connected fully to the next layer in the network.

LLM - Transformer based using 'attention' instead of 'activation' functions.  

# Size
MLP - Common to have up thousands of nodes.  This allows creation and training on relatively modest hardware.  Can even be restricted to CPU usage.

LLM - Even basic LLMs routinely have billions of nodes.  Except for the smallest LLMs, dedicated GPU/NN hardware is required to run.  For training the size requires distributed access to a grid of network hardware (GPUs or specialized CPUs)

# Purpose
MLP - Primarily used for classification and prediction given a set of inputs.

LLM - Designed for generative language tasks and generation.  Requires context/memory.

# Training
MLP - Gradient descent with loss function.

LLM - Similar to Gradient Descent with a "foundation model".
