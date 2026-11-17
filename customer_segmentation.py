
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Dataset
data = pd.read_csv("Mall_Customers.csv")

print("First 5 Rows of Dataset:")
print(data.head())

# Select Features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply K-Means Clustering

# Elbow Method
wcss = []

for i in range(1, 11):
    model = KMeans(n_clusters=i, random_state=42, n_init=10)
    model.fit(X)
    wcss.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.savefig("elbow_method.png")
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
from sklearn.metrics import silhouette_score
data['Cluster'] = kmeans.fit_predict(X)
# Calculate Silhouette Score
score = silhouette_score(X, data['Cluster'])

print("\nSilhouette Score:", score)
# Display Customer Groups
print("\nCustomer Segmentation Results:")
print(data[['CustomerID', 'Annual Income (k$)',
            'Spending Score (1-100)', 'Cluster']].head(10))

# Visualize Clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=data['Cluster'],
    cmap='rainbow',
    s=50
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    color='black',
    marker='X',
    s=200,
    label='Centroids'
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.savefig("customer_clusters.png")
plt.show()

print("\nCustomer Segmentation Completed Successfully!")
