# Dimensionality Reduction and Clustering on Handwritten Digits Dataset

## Research Overview
This research project explores the use of **dimensionality reduction** and **unsupervised learning** algorithms on the `digits` dataset, sourced from the `sklearn` library. The dataset consists of 64 features (pixel intensity values) for each image, representing handwritten digits (0-9). The objective of this research is to reduce the dimensionality of the dataset and apply clustering techniques to group similar images without using the original labels.

For more details on the dataset, refer to the [Scikit-learn Digits Dataset Documentation](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html).

### Objective
The primary goal is to cluster the digits into three distinct classes using two clustering algorithms:
1. **K-Means Clustering**
2. **Gaussian Mixture Models (GMM)**

The clustering results will then be compared to the true labels using evaluation metrics such as adjusted Rand index, homogeneity score, or silhouette score.

## Methodology
### Dataset Description
- The `digits` dataset contains 64 features for each image, each having a value between 0 and 16.
- The target variable, `y`, has 10 distinct classes (0-9), corresponding to the digits.

### Research Approach
1. **Dimensionality Reduction**  
   Perform dimensionality reduction on the dataset to reduce it to **two features** using:
   - **PCA (Principal Component Analysis)** or
   - **t-SNE (t-Distributed Stochastic Neighbor Embedding)**

   This step is crucial for visualizing the clustering results and simplifying the data structure for subsequent clustering.

2. **Selection of 3 Digits for Analysis**  
   Randomly select 3 different digits (e.g., 0, 3, and 7) from the dataset, which will serve as the basis for three distinct classes.

3. **K-Means Clustering**  
   - Run the **K-Means algorithm** on the reduced dataset for values of `k = 1, 2, ..., 10`.
   - Use **at least two different initial conditions** to validate the stability of the clustering.
   - Determine the optimal `k` using the **Elbow Method**, aiming to identify 3 clusters.

4. **Gaussian Mixture Model (GMM)**  
   - Initialize the covariance matrix (`sigma`) as a **2x2 diagonal matrix**, with diagonal elements equal to the variance of each feature.
   - Implement the **log-sum-exp trick** to handle numerical stability issues during the log-likelihood calculations.
   - Run the GMM algorithm and track the decrease in the cost function to ensure convergence.

5. **Comparison with True Labels**  
   Compare the clustering results with the true class labels using evaluation metrics such as:
   - **Adjusted Rand Index**
   - **Homogeneity Score**
   - **Silhouette Score**
