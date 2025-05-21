# Emotion Recognition from Movie Footage

This repository contains a deep learning project focused on recognizing emotions from image data extracted from raw movie footage. The project addresses the challenge of data scarcity by implementing a comprehensive data pipeline and augmentation strategies.

## Project Overview

This project involves building and evaluating a deep learning model for emotion recognition. A key aspect of this work is the meticulous preparation of a custom image dataset from raw video, followed by the implementation of a convolutional neural network (CNN) from scratch and analysis using clustering techniques.

## Key Features

* **Custom Data Pipeline:** Engineered a robust data pipeline to construct a novel image dataset from raw movie footage, including:

  * Frame extraction

  * Format standardization

  * Image normalization

  * De-duplication for high-quality inputs

* **Data Augmentation:** Strategically applied geometric transformations (rotation, cropping, flipping) to significantly increase dataset volume and variability, enhancing model robustness and generalization.

* **Dataset Management:** Structured and versioned the curated dataset into training (70%), validation (20%), and testing (10%) splits.

* **Efficient Data Loading:** Developed efficient data loading mechanisms using TensorFlow's `image_dataset_from_directory` for streamlined model development and evaluation.

* **Custom CNN Implementation:** Implemented a basic Convolutional Neural Network (CNN) architecture from scratch, including:

  * Convolutional Layers

  * Pooling Layers (Max/Average)

  * ReLU Activation

  * Predefined filters for feature extraction.

* **Feature Extraction & Clustering:** Extracted features using the custom CNN and applied K-Means clustering for unsupervised analysis of the image data.

* **Clustering Evaluation & Visualization:** Evaluated clustering performance using Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI), and visualized clusters using PCA.

## Technologies Used

* Python

* TensorFlow

* NumPy

* PIL (Pillow)

* Matplotlib

* Scikit-learn (for K-Means and evaluation metrics)

## How to Run

The core of this project is presented in a Jupyter Notebook (`DeepLearningProject.ipynb`).

1. **Clone the repository:**

git clone https://github.com/YourUsername/Emotion-Recognition-Deep-Learning.git
cd Emotion-Recognition-Deep-Learning

2. **Prepare the dataset:** The notebook expects a zipped dataset (`DLCV_SS25_Dataset.zip`) to be available, which it then unzips and processes. Ensure you have access to this dataset and place it as indicated in the notebook (e.g., in your Google Drive if running in Colab).

3. **Open and run the notebook:**

* **Google Colab:** Upload `DeepLearningProject.ipynb` to Google Colab and run all cells.

* **Jupyter Notebook/Lab:** Ensure you have all dependencies installed (`pip install tensorflow numpy pillow matplotlib scikit-learn`) and run the notebook locally.

## Future Work

* Integrate a pre-trained model (e.g., VGG, ResNet) for transfer learning to improve emotion recognition accuracy.

* Experiment with more advanced data augmentation techniques.

* Implement and train a full end-to-end deep learning model for classification.

* Explore different clustering algorithms or dimensionality reduction techniques.
