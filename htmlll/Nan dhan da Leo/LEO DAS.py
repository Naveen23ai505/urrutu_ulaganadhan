1.DECISION TREE ALGORITHM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define the data as a dictionary
data = {
    "Day": ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14"],
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
    "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])
X = df_encoded.drop(['Day', 'PlayTennis'], axis=1)
y = df['PlayTennis'].map({'Yes': 1, 'No': 0})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train the model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))



**************************************************************************************************************************************************


2.K MEANS CLUSTERING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the data as a dictionary
data = {
    "Individual": [1, 2, 3, 4, 5, 6, 7],
    "Variable 1": [1.0, 1.5, 3.0, 5.0, 3.5, 4.5, 3.5],
    "Variable 2": [1.0, 2.0, 4.0, 7.0, 5.0, 5.0, 4.5]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Extract features
X = df[['Variable 1', 'Variable 2']].values

# Step 1: Initialize centroids (randomly choose two points from the dataset as initial centroids)
np.random.seed(0)
initial_centroids = X[np.random.choice(X.shape[0], 2, replace=False)]
centroids = initial_centroids.copy()

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Step 2: K-means clustering iterations
for iteration in range(10):  # Run for a fixed number of iterations
    # Step 3: Assign each point to the nearest centroid
    clusters = {}
    for i in range(len(centroids)):
        clusters[i] = []
    
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        clusters[closest_centroid].append(point)
    
    # Step 4: Update centroids by calculating the mean of points in each cluster
    new_centroids = []
    for i in range(len(centroids)):
        if clusters[i]:  # Avoid empty clusters
            new_centroid = np.mean(clusters[i], axis=0)
            new_centroids.append(new_centroid)
        else:
            new_centroids.append(centroids[i])  # Keep previous centroid if cluster is empty
    
    new_centroids = np.array(new_centroids)
    
    # Check if centroids have changed; if not, break out of loop
    if np.all(centroids == new_centroids):
        break
    centroids = new_centroids

# Display the final centroids and cluster assignment
for i, cluster in clusters.items():
    print(f"Cluster {i+1}: {cluster}")

# Plot the clusters and centroids
for i, cluster in clusters.items():
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i+1}')

plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', label='Centroids', marker='X')
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.title('K-means Clustering')
plt.legend()
plt.show()


**************************************************************************************************************************************************


3LINEAR REGRESSION

# Given data points
x_values = [0, 1234]
y_values = [2, 3546]
n = len(x_values)

# Calculating necessary sums
sum_x = sum(x_values)
sum_y = sum(y_values)
sum_x_squared = sum([x**2 for x in x_values])
sum_xy = sum([x_values[i] * y_values[i] for i in range(n)])

# Calculating slope (a) and intercept (b)
a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
b = (sum_y - a * sum_x) / n

# Estimating y when x = 10
x_estimate = 10
y_estimate = a * x_estimate + b

# Calculating errors for each data point and Mean Squared Error (MSE)
errors = [y_values[i] - (a * x_values[i] + b) for i in range(n)]
mse = sum([error**2 for error in errors]) / n

a, b, y_estimate, errors, mse


**************************************************************************************************************************************************


4 S ALGORITHM
data = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"]
]
hypothesis = ["0", "0", "0", "0", "0", "0"]
for instance in data:
    if instance[-1] == "Yes":  # Consider only positive examples
        for i in range(len(hypothesis)):
            if hypothesis[i] == "0":  
                hypothesis[i] = instance[i]
            elif hypothesis[i] != instance[i]:  
                hypothesis[i] = "?"
print("Maximally Specific Hypothesis:", hypothesis)

**************************************************************************************************************************************************

5. GMM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Using only the first two features for easy visualization

# GMM Hyperparameters
n_components = 3  # Number of clusters
max_iter = 100  # Max number of iterations
tol = 1e-6  # Tolerance for convergence

# Initialize parameters
np.random.seed(42)
n_samples, n_features = X.shape

# Initialize means randomly from data points
means = X[np.random.choice(n_samples, n_components, replace=False)]

# Initialize covariances as identity matrices
covariances = np.array([np.eye(n_features)] * n_components)

# Initialize weights equally
weights = np.ones(n_components) / n_components

# Function to calculate multivariate Gaussian density
def gaussian_pdf(X, mean, cov):
    d = X.shape[1]  # Number of features
    diff = X - mean
    exponent = np.sum(diff @ np.linalg.inv(cov) * diff, axis=1)
    return np.exp(-0.5 * exponent) / (np.sqrt((2 * np.pi) ** d * np.linalg.det(cov)))

# Expectation-Maximization (EM) Algorithm
log_likelihoods = []
for _ in range(max_iter):
    # E-step: Compute responsibilities (probabilities)
    resp = np.zeros((n_samples, n_components))
    for i in range(n_components):
        resp[:, i] = weights[i] * gaussian_pdf(X, means[i], covariances[i])
    
    # Normalize responsibilities
    resp /= resp.sum(axis=1, keepdims=True)
    
    # M-step: Update parameters
    N_k = resp.sum(axis=0)
    
    # Update means
    means = (resp.T @ X) / N_k[:, np.newaxis]
    
    # Update covariances
    for i in range(n_components):
        diff = X - means[i]
        covariances[i] = (resp[:, i] * diff).T @ diff / N_k[i]
    
    # Update weights
    weights = N_k / n_samples
    
    # Compute log-likelihood
    log_likelihood = np.sum(np.log(resp.sum(axis=1)))
    log_likelihoods.append(log_likelihood)
    
    # Check for convergence
    if len(log_likelihoods) > 1 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
        break

# Assign labels to the data points
labels = np.argmax(resp, axis=1)

# Plot the results
colors = ['red', 'green', 'yellow']
for i in range(n_components):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], color=colors[i], label=f'Gaussian {i + 1}')

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Gaussian Mixture Model on Iris Dataset')
plt.legend()
plt.show()



**************************************************************************************************************************************************

6 SVM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Given Data Points (Corrected)
X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11], [7, 10], [8.7, 9.4],
              [2.3, 4], [5.5, 3], [7.7, 8.8], [6.1, 7.5]])
y = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1])  # Fixed y array length

# Train SVM classifier with a linear kernel
svm = SVC(kernel='linear', C=1000)
svm.fit(X, y)

# Get the coefficients (weights) and intercept for the hyperplane
w = svm.coef_[0]
b = svm.intercept_[0]

# Plotting the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', marker='o', s=100, edgecolors='k')

# Plotting the optimal hyperplane (decision boundary)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and margins
plt.contour(xx, yy, Z, levels=[-1, 0, 1], linewidths=[1, 2, 1], colors=['orange', 'black', 'orange'])
plt.title('SVM with Optimal Hyperplane and Marginal Planes')

# Show the plot
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()



*********************************************************************************************************************************************
7.KNN
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Given data points
data = [
    [158, 58, 'M'], [158, 59, 'M'], [158, 63, 'M'], [160, 59, 'M'], [160, 60, 'M'],
    [163, 60, 'M'], [163, 61, 'M'], [160, 64, 'L'], [163, 64, 'L'], [165, 61, 'L'],
    [165, 62, 'L'], [165, 65, 'L'], [168, 62, 'L'], [168, 63, 'L'], [168, 66, 'L'],
    [170, 63, 'L'], [170, 64, 'L'], [170, 68, 'L']
]

# Convert data to numpy array for easier handling
data = np.array(data)

# Extract heights, weights and labels
heights = data[:, 0].astype(float)
weights = data[:, 1].astype(float)
labels = data[:, 2]

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return np.sqrt(distance)

# K-Nearest Neighbors function (without using inbuilt functions)
def k_nearest_neighbors(new_point, k=3):
    distances = []
    
    # Calculate distance to each data point
    for i in range(len(data)):
        dist = euclidean_distance(new_point, np.array([heights[i], weights[i]]))
        distances.append((dist, labels[i]))

    # Manually sort the distances
    distances.sort(key=lambda x: x[0])

    # Select k nearest neighbors
    nearest_neighbors = distances[:k]
    
    # Get the labels of the k nearest neighbors
    neighbor_labels = [neighbor[1] for neighbor in nearest_neighbors]
    
    # Count the frequency of each label
    label_count = Counter(neighbor_labels)
    
    # Return the most common label
    prediction = label_count.most_common(1)[0][0]
    return prediction

# New customer data
new_customer = np.array([161, 61])

# Predict the T-shirt size for the new customer
predicted_size = k_nearest_neighbors(new_customer, k=3)
print(f"The predicted T-shirt size for the new customer is: {predicted_size}")

# Plotting the data points
plt.scatter(heights[labels == 'M'], weights[labels == 'M'], color='r', label='M')
plt.scatter(heights[labels == 'L'], weights[labels == 'L'], color='b', label='L')
plt.scatter(new_customer[0], new_customer[1], color='g', marker='x', label='New Customer')
plt.xlabel('Height (in cms)')
plt.ylabel('Weight (in kgs)')
plt.legend()
plt.title('K-Nearest Neighbors')
plt.show()

**************************************************************************************************************************************************
