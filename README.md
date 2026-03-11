# Neural Networks from scratch

This repository contains an implementation of a modular neural network framework built in Python.  
The goal of the project is to understand how neural networks operate internally by implementing core deep learning components from scratch, including layers, loss functions, optimizers, and training pipelines.

Instead of relying entirely on existing deep learning libraries, this project separates each major component of a neural network system into its own module. This makes it easier to experiment with architectures, training strategies, and optimization methods.

---

## Repository Structure

### data
Contains datasets and code responsible for loading and preparing data used for training and evaluation.

### data_scalers
Implements data normalization and scaling techniques that prepare input features for neural network training.

### layers
Defines the building blocks of neural networks such as fully connected, recurrent, convolutional layers and activation functions.

### loss_functions
Contains implementations of loss functions used to measure the difference between model predictions and true labels during training.

### metrics
Provides evaluation metrics used to measure model performance during and after training.

### models
Defines complete neural network architectures constructed using the available layers.

### optimizers
Implements optimization algorithms used to update model parameters during training.

### saved_models
Stores trained neural network models so they can be reused without retraining.

### test_examples
Example scripts used to test different components of the framework and demonstrate how models can be trained and evaluated.

### testing_saved_models
Contains scripts used to load previously trained models and evaluate their performance.

### utils
Utility functions that support different parts of the project, such as data handling, logging, or helper operations.

### weight_initializers
Implements different strategies for initializing neural network weights before training begins.

---

## Main Files

### main.py
Entry point for running training or testing workflows using the implemented neural network framework.

### requirements.txt
Lists the Python dependencies required to run the project.

### Wildcat Species Image Classification
Example project demonstrating how the framework can be used for an image classification task.

---

## Technologies

- Python  
- NumPy  
- Machine Learning  
- Neural Networks
