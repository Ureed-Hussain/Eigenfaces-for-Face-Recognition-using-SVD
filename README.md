# 👤 Eigenfaces for Face Recognition using SVD (MATLAB)

This project demonstrates the classical **Eigenfaces** approach for face recognition using Singular Value Decomposition (SVD). The pipeline includes preprocessing, visualization, dimensionality reduction via PCA, face reconstruction, and classification based on nearest-neighbor search in eigenspace.

## 🧠 Key Concepts

- Principal Component Analysis (PCA)
- Singular Value Decomposition (SVD)
- Eigenfaces generation
- Dimensionality reduction
- Face reconstruction
- Classification using Euclidean distance in eigenspace

## 📁 Dataset

The dataset (`allFaces.mat`) contains facial images of multiple persons. Each image is vectorized and stacked as columns in the `faces` matrix. The `nfaces` array holds the number of images per person.

## 🔧 Dependencies

- MATLAB
- `allFaces.mat` file in `../codefile/`

---

## 🚀 How It Works

### 1. **Data Visualization**

- Displays all face images in a grid layout.
- Also visualizes individual persons' face sets in an 8×8 arrangement.

### 2. **Training Setup**

- First 36 persons' images are used for training.
- The mean face (`avgFace`) is computed and visualized.
- SVD is performed on the mean-subtracted training data to generate the eigenfaces.

### 3. **Eigenfaces Visualization**

- Top eigenfaces (principal components) are reshaped and displayed in an 8×8 layout.

### 4. **Face Reconstruction**

- Reconstructs a test face using varying numbers of eigenfaces (`r` = 4, 10, 20, 100, ...).
- Plots side-by-side comparisons of original and reconstructed faces.

### 5. **Dimensionality Reduction and Projection**

- Projects images of two selected persons (Person 2 and Person 7) onto PCA dimensions 5 and 6.
- Plots these 2D projections for visual class separation.

### 6. **Face Recognition**

- Uses top 100 principal components.
- For each test image (from persons 37 onward), performs classification via nearest neighbor search in PCA space.
- Computes **recognition accuracy**.

### 7. **RMSE Calculation**

- Computes the **Root Mean Square Error (RMSE)** between original and reconstructed faces for varying `r`.

### 8. **Reconstruction Comparison**

- Displays original vs. reconstructed faces for Person 2 and Person 7 using only PC5 and PC6.

---

## 📊 Results

- Recognition accuracy is printed.
- RMSE for each value of `r` is displayed as a table:

