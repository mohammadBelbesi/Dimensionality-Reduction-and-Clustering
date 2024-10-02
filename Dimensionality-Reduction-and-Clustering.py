import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

# Load digits dataset
digits = datasets.load_digits()
X, y = digits.data, digits.target

# Select 3 digits (for example: 0, 1, 2)
selected_classes = [0, 1, 2]
indices = np.isin(y, selected_classes)
X, y = X[indices], y[indices]

# Reduce dimensionality to 2 using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Reduce dimensionality to 2 using TSNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Function to plot the Elbow Method
def plot_elbow_method(X, max_k=10):
    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    
    plt.plot(range(1, max_k + 1), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method For Optimal k')
    plt.show()

# Elbow method plot for PCA
plot_elbow_method(X_pca)

# Elbow method plot for TSNE
plot_elbow_method(X_tsne)

# Run KMeans with 3 clusters on PCA
kmeans_pca = KMeans(n_clusters=3, random_state=42)
labels_pca = kmeans_pca.fit_predict(X_pca)

# Run KMeans with 3 clusters on TSNE
kmeans_tsne = KMeans(n_clusters=3, random_state=42)
labels_tsne = kmeans_tsne.fit_predict(X_tsne)

# Display KMeans results
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_pca, cmap='viridis', s=50)
plt.title('KMeans Clustering on PCA Reduced Data')

plt.subplot(122)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_tsne, cmap='viridis', s=50)
plt.title('KMeans Clustering on TSNE Reduced Data')

plt.show()

# Compare KMeans results to true labels using adjusted_rand_score
print("Adjusted Rand Score for PCA (KMeans):", adjusted_rand_score(y, labels_pca))
print("Adjusted Rand Score for TSNE (KMeans):", adjusted_rand_score(y, labels_tsne))

# Initialize GMM with 3 clusters
gmm_pca = GaussianMixture(n_components=3, covariance_type='diag', random_state=42)
gmm_tsne = GaussianMixture(n_components=3, covariance_type='diag', random_state=42)

# Train GMM on PCA
gmm_pca.fit(X_pca)
labels_gmm_pca = gmm_pca.predict(X_pca)

# Train GMM on TSNE
gmm_tsne.fit(X_tsne)
labels_gmm_tsne = gmm_tsne.predict(X_tsne)

# Display GMM results
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_gmm_pca, cmap='viridis', s=50)
plt.title('GMM Clustering on PCA Reduced Data')

plt.subplot(122)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_gmm_tsne, cmap='viridis', s=50)
plt.title('GMM Clustering on TSNE Reduced Data')

plt.show()

# Compare GMM results to true labels using adjusted_rand_score
print("Adjusted Rand Score for PCA (GMM):", adjusted_rand_score(y, labels_gmm_pca))
print("Adjusted Rand Score for TSNE (GMM):", adjusted_rand_score(y, labels_gmm_tsne))