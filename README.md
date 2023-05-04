# MNIST-Handwritten-Digit-Recognition-TensorFlow

This repository contains a TensorFlow implementation of a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset.

## Data Preparation
The MNIST dataset is loaded from TensorFlow's datasets, and the data is preprocessed, including:

* Converting the dataset to a Pandas DataFrame.
* Scaling the image pixel values by dividing them by 255.
* Stratified shuffling to ensure a balanced distribution of labels in the train, validation, and test sets.

## Feature Engineering
The input images are used directly without any specific feature engineering, as CNNs are capable of learning the important features automatically.

## CNN Implementation
The CNN model architecture consists of the following layers:

* Three convolutional layers, each followed by a max-pooling layer, to learn spatial features from the input images.
* Two dense layers to learn more complex features and patterns.
* An output layer with 10 units and a softmax activation function for multi-class classification.

## Model Building
The model is created using TensorFlow's Keras API, with the layers defined sequentially. The model is compiled using the Adam optimizer, sparse categorical crossentropy loss, and accuracy metric.

## Training and Evaluation
The model is trained on the preprocessed data using the fit() method, with early stopping to prevent overfitting. The model is evaluated on the test dataset to measure its performance.

## Results Visualization
After training, the following visualizations are generated:

* Loss and accuracy curves, showing the training and validation loss and accuracy over time.
* Confusion matrix, comparing true labels to predicted labels for each class.
* Feature maps, displaying the activations of the first convolutional layer for a single input image.

## Usage
To use this code, simply run the provided Python script. Ensure that you have the required libraries installed, including TensorFlow, NumPy, pandas, matplotlib, seaborn, and scikit-learn.

## Conclusion
This implementation demonstrates the effectiveness of CNNs in recognizing handwritten digits using the MNIST dataset. The model achieves high accuracy on both training and validation sets, and the visualizations provide insights into the model's performance and learned features.

