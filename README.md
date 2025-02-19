# CIFAR-100 CNN Classifier

This project is a Convolutional Neural Network (CNN) model built using TensorFlow and Keras to classify images from the CIFAR-100 dataset. The dataset consists of 100 different object classes, each containing 60000 images.

## Features
- Loads and preprocesses the CIFAR-100 dataset
- Resizes images from 32x32 to 64x64 resolution
- Visualizes sample images from the dataset
- Defines a CNN model using TensorFlow/Keras
- Trains the model with optimized parameters
- Evaluates model performance using accuracy and loss metrics
- Saves and loads the trained model

## Installation

To run this project, you need to have Python installed along with the required dependencies. You can install them using:

```bash
pip install tensorflow numpy matplotlib
```

## Usage

1. Clone this repository or download the notebook file.
2. Run the Jupyter Notebook step by step to:
   - Load and preprocess data
   - Train the CNN model
   - Evaluate and visualize results
3. Modify model parameters as needed to experiment with different architectures.

## Dataset
The CIFAR-100 dataset contains 100 classes, each with 60000 images (500 training images and 100 test images per class). The images are resized from 32x32 pixels to 64x64 pixels with 3 color channels.

## Model Architecture
The CNN model consists of:
- Convolutional layers with ReLU activation
- Batch normalization and dropout layers
- Max-pooling layers for dimensionality reduction
- A fully connected output layer with softmax activation

## Results
The model achieved:
- **Training Accuracy:** 84%
- **Validation Accuracy:** 67%

Performance can be further improved by tuning hyperparameters or using data augmentation techniques.

## License
This project is open-source and free to use.
