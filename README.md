# Multi-Layer Perceptron Implementation for MNIST

This project implements a Multi-Layer Perceptron (MLP) from scratch using PyTorch for training and evaluating on the MNIST dataset. The objective is to explore various optimization techniques and regularization methods to improve model performance. The project includes multiple experiments with different configurations and hyperparameters to analyze their impact on training time and accuracy.

## Objective

The task involves implementing an MLP and experimenting with the following techniques:

1. **Gradient Descent** - Basic gradient descent for training the model.
2. **Momentum** - Enhanced gradient descent with momentum to speed up convergence.
3. **Optimizers** - Testing built-in PyTorch optimizers such as SGD and Adam.
4. **Regularization Techniques** - Experimenting with L2 regularization and dropout to prevent overfitting.
5. **Mini-Batch Gradient Descent** - Training with mini-batches and varying batch sizes to optimize training speed and stability.

## Dataset

The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9). It is commonly used for image classification tasks.

## Features

- **MLP Architecture**: 
  - Input layer: 784 inputs (28x28 flattened images).
  - Hidden layer(s): At least one hidden layer with ReLU activation function.
  - Output layer: 10 outputs (one for each class in MNIST).

- **Optimization Techniques**:
  - Implemented gradient descent with and without momentum.
  - Experiments with PyTorch's SGD and Adam optimizers.

- **Regularization**:
  - L2 regularization and dropout implemented to reduce overfitting.

- **Mini-Batch Training**:
  - Modified training to use mini-batches with varying batch sizes to compare effects on training time and accuracy.

## Experimentation

- **Momentum**: Helps accelerate gradient descent in the correct direction, reducing oscillations.
- **Optimizers**: Comparison of SGD and Adam optimizers to explore their effects on convergence speed and accuracy.
- **Regularization**: L2 regularization and dropout applied to observe their impact on overfitting and generalization.
- **Batch Sizes**: Mini-batch training was tested with different batch sizes (e.g., 32, 64, 128) to analyze the effect on training speed and model accuracy.

## Results

The project includes plots showing:
- Training time and epochs to convergence.
- Accuracy on the test set for different optimizers, batch sizes, and regularization techniques.

## Requirements

- Python 3.x
- PyTorch
- Matplotlib
- NumPy

## Evaluation

The project was evaluated based on:
- Correctness of the MLP implementation and training setup.
- Experimentation with all variations (momentum, optimizers, regularization, batch sizes).
- Observations and analysis of how each change affects the model's performance.
- Code readability and documentation.


