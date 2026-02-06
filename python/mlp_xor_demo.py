"""
MLP Neural Network XOR Classification Demo

This script demonstrates a Multi-Layer Perceptron (MLP) neural network
using scikit-learn to learn the XOR function.

Architecture: 2-2-1 (2 inputs, 2 hidden neurons, 1 output)
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Set random seed for reproducibility
np.random.seed(42)

# Create XOR dataset
# XOR truth table:
# Input 1 | Input 2 | Output
#    0    |    0    |   0
#    0    |    1    |   1
#    1    |    0    |   1
#    1    |    1    |   0

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 1, 0])

print("=" * 60)
print("MLP Neural Network XOR Classification Demo")
print("=" * 60)
print(f"\nTraining Data (XOR):")
print("Input 1 | Input 2 | Expected Output")
print("-" * 35)
for i in range(len(X)):
    print(f"   {X[i][0]}    |    {X[i][1]}    |       {y[i]}")

# Create MLPClassifier with 2-2-1 architecture
# hidden_layer_sizes=(2,) means 1 hidden layer with 2 neurons
# activation='tanh' or 'relu' work well for XOR
# solver='lbfgs' is good for small datasets
# max_iter increased to ensure convergence
# early_stopping=False to train fully
# tol (tolerance) set lower for better convergence

print("\n" + "=" * 60)
print("Training the MLP Neural Network...")
print("=" * 60)

# Train with retry mechanism to ensure 100% accuracy
max_attempts = 10
mlp = None
for attempt in range(max_attempts):
    mlp = MLPClassifier(
        hidden_layer_sizes=(2,),  # 2 hidden neurons (2-2-1 architecture)
        activation='tanh',         # Hyperbolic tangent activation
        solver='lbfgs',            # L-BFGS optimizer (good for small datasets)
        max_iter=10000,            # Maximum iterations (increased)
        random_state=42 + attempt, # Vary random seed if retrying
        learning_rate_init=0.1,    # Initial learning rate
        early_stopping=False,      # Don't stop early
        tol=1e-6,                  # Lower tolerance for better convergence
        warm_start=False           # Start fresh each time
    )
    
    # Train the data.    Forward pass, Loss calc, Backproagation, weight updates All of this is in a single call to `.fit()`
    mlp.fit(X, y)
    
    # Check if we achieved 100% accuracy
    predictions = mlp.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    if accuracy == 1.0:
        if attempt > 0:
            print(f"Successfully trained after {attempt + 1} attempt(s)")
        break
    else:
        print(f"Attempt {attempt + 1}: Accuracy = {accuracy * 100:.1f}%, retrying...")

print(f"\nTraining completed!")
print(f"Number of iterations: {mlp.n_iter_}")
print(f"Loss: {mlp.loss_:.6f}")

# Test the model on all XOR combinations
print("\n" + "=" * 60)
print("Testing the trained model:")
print("=" * 60)
print("Input 1 | Input 2 | Expected | Predicted | Correct")
print("-" * 55)

# Recalculate predictions for display (in case we're not in retry loop)
predictions = mlp.predict(X)
for i in range(len(X)):
    correct = "✓" if predictions[i] == y[i] else "✗"
    print(f"   {X[i][0]}    |    {X[i][1]}    |    {y[i]}    |     {predictions[i]}     |   {correct}")

# Calculate accuracy
accuracy = accuracy_score(y, predictions)
print(f"\nAccuracy: {accuracy * 100:.1f}%")

# Display model architecture details
print("\n" + "=" * 60)
print("Model Architecture Details:")
print("=" * 60)
print(f"Input layer: {X.shape[1]} neurons")
print(f"Hidden layer: {mlp.hidden_layer_sizes[0]} neurons")
print(f"Output layer: {len(mlp.classes_)} neuron(s)")
print(f"\nWeights and biases:")
print(f"\nInput -> Hidden layer weights:")
print(mlp.coefs_[0])
print(f"\nHidden layer biases:")
print(mlp.intercepts_[0])
print(f"\nHidden -> Output layer weights:")
print(mlp.coefs_[1])
print(f"\nOutput layer bias:")
print(mlp.intercepts_[1])

# Visualize the decision boundary
print("\n" + "=" * 60)
print("Creating visualization...")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 6))

# Create a mesh to plot the decision boundary
h = 0.01  # Step size in the mesh
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict for each point in the mesh
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3)

# Plot the training points
colors = ['red', 'green']
for i in range(len(X)):
    ax.scatter(X[i, 0], X[i, 1], c=colors[y[i]], s=100, 
               edgecolors='black', linewidth=2, 
               label=f'Class {y[i]}' if i < 2 else '')

ax.set_xlabel('Input 1', fontsize=12)
ax.set_ylabel('Input 2', fontsize=12)
ax.set_title('MLP Neural Network XOR Classification\n(2-2-1 Architecture)', fontsize=14)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('mlp_xor_decision_boundary.png', dpi=150, bbox_inches='tight')
print("Visualization saved as 'mlp_xor_decision_boundary.png'")
plt.show()

print("\n" + "=" * 60)
print("Demo completed successfully!")
print("=" * 60)
