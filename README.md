# Face Recognition with PCA (Eigenfaces) and an ANN

Implements the eigenfaces method from scratch with NumPy, then classifies the compressed face representations with a multi-layer perceptron.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

## Problem

A 100×100 grayscale face is a 10,000-dimensional vector, far too many dimensions for 450 training images. PCA finds the few directions ("eigenfaces") that capture most of the variation between faces, so a classifier can work on 20 numbers per face instead of 10,000.

## Dataset

450 face images of **9 people, 50 images each**, from the course dataset at [robaita/introduction_to_machine_learning](https://github.com/robaita/introduction_to_machine_learning). The notebook downloads it automatically.

## Method

PCA is implemented by hand rather than with `sklearn.decomposition.PCA`:

1. **Preprocess:** read each image as grayscale, resize to 100×100, flatten. The data matrix `X` is 10,000 × 450.
2. **Mean face:** subtract the average face from every image.
3. **Covariance trick:** instead of the 10,000 × 10,000 covariance matrix, compute the 450 × 450 matrix `Δᵀ Δ`. It has the same non-zero eigenvalues and is far cheaper to decompose.
4. **Eigen-decomposition:** `np.linalg.eig`. Sort by eigenvalue and keep the top **K = 20** eigenvectors.
5. **Eigenfaces:** project the eigenvectors back into image space, giving 20 eigenfaces of 10,000 pixels each.
6. **Face signatures:** project every face onto the eigenfaces, giving a 20-number signature per image.
7. **Classify:** `MLPClassifier` (one hidden layer of 100 units) on a 60/40 train/test split.
8. **Imposter check:** a test face is labelled *unknown* if the classifier's top probability is below 0.6.

The notebook also sweeps K ∈ {5, 10, 15, 20, 25} and plots accuracy against K, and it visualizes the first 10 eigenfaces.

## Results

| | |
|---|---|
| Test accuracy (saved notebook run) | **40.6%** |
| Re-runs (verified, 3 runs) | 44.4% to 49.4% |
| Random-guess baseline (9 classes) | 11.1% |

The model learns real structure, at about four times chance, but accuracy is modest. Two reasons: eigenfaces capture lighting and pose as much as identity, and 30 training images per person is very little data for an MLP. Results also vary between runs because the MLP is not seeded.

## Run it

```bash
git clone https://github.com/sahithi-manda/PCA-ANN-Face-Recognition.git
cd PCA-ANN-Face-Recognition
pip install -r requirements.txt
jupyter notebook PCA_ANN_FaceRecognition.ipynb
```

The notebook was written in Google Colab. Its download cell uses `wget`. If you run it locally without `wget`, download `dataset.zip` from the link above and unzip it next to the notebook.

## Files

```
├── PCA_ANN_FaceRecognition.ipynb   # Full pipeline with outputs
├── project_lyst7052.pdf            # Original assignment brief
└── requirements.txt
```

## Next steps

- Standardize the face signatures before the MLP and fix `random_state` so runs are reproducible.
- Use a stratified split and report a confusion matrix to see which identities get confused.
- Compare against an SVM and k-NN on the same signatures, and against `sklearn.decomposition.PCA(whiten=True)`.
- Hold out a separate image for the inference demo. It currently uses an image from the dataset.
