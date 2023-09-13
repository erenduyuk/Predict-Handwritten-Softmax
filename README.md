# Handwritten Digit Classification with Neural Network

This Python script demonstrates how to build a neural network model to classify handwritten digits using the MNIST dataset. It utilizes TensorFlow and the Keras library for model creation and training.

## Getting Started

These instructions will help you run the code on your local machine for development and testing purposes.

### Prerequisites

Before running the code, make sure you have the following libraries installed:

- numpy
- pandas
- tensorflow
- matplotlib
- mnist (for dataset download)


## Dataset

The MNIST dataset is used for this project, containing handwritten digit images. It is automatically downloaded using the mnist library.

## Data Preprocessing

- The dataset is normalized to ensure that pixel values are in the range [0, 1].
- A neural network model with layers for flattening, dense, and softmax activation is created.

## Model

The neural network model is constructed with the following layers:

- Flatten Layer: To flatten the 28x28 pixel images.
- Dense Layer (128 units): Using ReLU activation.
- Dense Layer (128 units): Using ReLU activation.
- Output Dense Layer (10 units): Using softmax activation for classification.

The model is compiled using categorical cross-entropy loss and the Adam optimizer. It is trained on the training data for a specified number of epochs.

## Model Evaluation

- Training and validation accuracy are tracked over epochs.
- The script demonstrates how to make predictions on individual images from the dataset.
- The overall accuracy on the test dataset is calculated and displayed.

## Results

- The script prints out the predicted class for an example image.
- It also shows the accuracy of the model on the test dataset.
- Train accuracy is 96.6

## License

This project is licensed under the MIT License

