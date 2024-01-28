# Convolutional Neural Networks for CIFAR-10 Image Classification

## Project Overview
This project involves building a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 different classes.

## Problem Description
- The goal is to develop a CNN model to classify CIFAR-10 images effectively.
- We will experiment with different numbers of convolutional and subsampling layers.
- The implementation is done using Keras or TensorFlow.
- The model will be trained on a 90/10 split of the CIFAR-10 dataset and evaluated using accuracy metrics, a confusion matrix, and ROC curve.

## Technical Approach
- **Preprocessing**: Input data will be reshaped and normalized using Keras preprocessing tools.
- **Model Building**: Constructing a baseline CNN model and then iterating with different configurations.
- **Evaluation**: Using a 90/10 data split for training and validation.
- **Visualization**: Accuracy and loss curves will be plotted for model analysis.
- **Hyperparameter Tuning**: Grid search will be employed for tuning.
- **Optimization Comparison**: Different optimization algorithms will be compared.
  
## Dependencies
To run this project, you will need the following libraries:

- TensorFlow: For building and training the CNN model.
- Keras: High-level neural networks API, running on top of TensorFlow.
- scikit-learn: For model evaluation and hyperparameter tuning.
- Matplotlib: For plotting graphs and visualizations.
- Numpy: For numerical and array operations.

## Dataset Analysis
- An analysis will be provided for the CIFAR-10 dataset, focusing on label counts and data distribution.


<p align="center">
  <img src="https://github.com/tanzealist/-CIFAR-10-Image-Classification-with-CNNs/assets/114698958/f7e0c45a-8b41-4bbe-971e-04390bfeb543" alt="Model Accuracy Plot" width="600">
</p>

<p align="center">
  <img src="https://github.com/tanzealist/-CIFAR-10-Image-Classification-with-CNNs/assets/114698958/b5b8054d-65ea-4362-a12b-948a7bdf3ede" alt="Model Loss Plot" width="600">
</p>



## CNN Models
- **Model 1**: 4 convolutional layers with max pooling and dropout, two dense layers.
- **Model 2**: 5 convolutional layers configuration.
- **Model 3**: 6 convolutional layers setup.
- **Model 4**: 6 convolutional layers with SGD optimizer.
- **Model 5**: 4 convolutional layers with hyperparameter tuning (batch size, epochs, dropout rate, optimizer).

## Final Results
- The model with 4 convolutional layers, max pooling, dropout, and specific hyperparameter tuning displayed the best accuracy.
- A detailed analysis of the best model is provided, including a confusion matrix exploration.

 <p align="center">
  <img src="https://github.com/tanzealist/-CIFAR-10-Image-Classification-with-CNNs/assets/114698958/2b3cf15d-e9fb-4764-b41e-b7fe28cc531c" alt="Description of New Image" width="600">
</p>

<p align="center">
  <img src="https://github.com/tanzealist/-CIFAR-10-Image-Classification-with-CNNs/assets/114698958/6868d2ab-f85d-478f-a540-f2f21d83bb6c" alt="Description of Previous Image" width="600">
</p>



## Conclusion

Our investigation into various CNN architectures for image classification on the CIFAR-10 dataset has led to significant insights:

- CNNs have proven highly effective, with our models achieving accuracy rates indicative of their robustness in image classification tasks.
- The architecture of the CNN, particularly the number of convolution layers, plays a crucial role in model performance.
- Through meticulous hyperparameter tuning, specifically adjusting learning rates and epochs, model accuracy is notably enhanced.
- The standout model, featuring four convolution layers coupled with strategic hyperparameter adjustments, attained a peak accuracy of 79.22%.

These findings underscore the critical impact of CNN architecture design and the benefits of hyperparameter optimization in advancing image classification endeavors.

## Additional Information
Run the Tanuj_cnn.ipynb file to run this project.To understand the flow of project refer the CNN_PPT.


