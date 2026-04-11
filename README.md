# Face Recognition using PCA and ANN

This repository contains a complete pipeline for a Face Recognition system built using Principal Component Analysis (PCA) and an Artificial Neural Network (ANN). 

By applying PCA, we reduce the high dimensionality of facial images to extract the most critical features—known as "Eigenfaces." These key features are then fed into an ANN (Multi-Layer Perceptron), which learns to accurately classify and recognize the faces of different individuals. This approach significantly boosts computational efficiency while maintaining high recognition accuracy.

## Features & Workflow

1. **Data Preprocessing**: Reads face images, resizes them to a fixed uniform size (100x100 pixels), converts them to grayscale, and flattens them into 1D vectors for mathematical operations.
2. **Mean Face & Covariance Matrix**: Calculates the "mean face" (average structure of all faces in the dataset) and normalizes the images by subtracting the mean. Then, computes the simplified covariance matrix.
3. **Eigenface Generation (PCA)**: Extracts eigenvalues and eigenvectors to find the directions of maximum variance in the dataset. The top *K* eigenvectors are selected to generate **Eigenfaces** and reduce the dimensionality of the images into compact "Face Signatures."
4. **ANN Classification**: Utilizes an `MLPClassifier` from `scikit-learn` to train a neural network on the extracted face signatures. 
5. **Evaluation**: Visualizes the effect of utilizing different values of *K* (number of principal components) on the model's overall accuracy.
6. **Inference**: Includes a script to load a test image, apply the same transformations (mean subtraction and subset projection), and predict the identity of the person using the trained ANN.
7. **Visualization**: Displays the generated "Eigenfaces" using `matplotlib` to analyze the features the model considers critical (e.g., eye regions, nose structures, and specific facial contours).

## Libraries Used
- **OpenCV (`cv2`)**: Image loading, resizing, and preparation.
- **NumPy (`numpy`)**: Mathematical operations, matrix computations, eigenvectors decomposition.
- **Scikit-Learn (`sklearn`)**: Splitting the dataset into train/test sets and training the `MLPClassifier`.
- **Matplotlib (`matplotlib`)**: Plotting accuracy graphs and visualizing the eigenfaces.

## Repository Contents
- `PCA_ANN_FaceRecognition.ipynb`: The main Jupyter Notebook containing the full implementation of the face recognition pipeline.
- `train_lyst1717074532669.csv` & `test_lyst1717074532669.csv`: Supplementary dataset files.
- `project_lyst7052.pdf`: Project documentation or instructions.
