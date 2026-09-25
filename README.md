# AI Internship – Task 03
# Handwritten Digit Recognition using Neural Networks

## Project Overview

This project is a handwritten digit recognition system built using Deep Learning and TensorFlow/Keras.

The model is trained on the MNIST dataset and can recognize handwritten digits from **0 to 9**.

The complete workflow is:

**Dataset → Preprocessing → Neural Network → Training → Evaluation → Prediction → Model Saving**

---

## 1. What is MNIST?

MNIST is a popular dataset used for handwritten digit recognition.

It contains:

- 60,000 training images
- 10,000 testing images
- 10 different classes (digits 0–9)
- Each image is 28 × 28 pixels
- Images are grayscale

Each image contains one handwritten digit.

For example:

```text
0  1  2  3  4
5  6  7  8  9

## Fine-Tuning

After the initial CNN was trained on the MNIST dataset, the saved model
was loaded and further trained using a smaller learning rate.

During the first fine-tuning stage, convolutional layers were frozen so
that the learned visual features were preserved while the upper layers
were adjusted.

In the second stage, convolutional layers were unfrozen and the model
was trained with a very small learning rate.

The fine-tuned model was saved as:

`mnist_handwritten_digit_fine_tuned.keras`

The final accuracy reported in the notebook is the actual measured
accuracy on the MNIST test dataset.