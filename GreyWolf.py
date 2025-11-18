# ------------------------------------------------------------
# Image Outline Deepening using Grey Wolf Optimization (GWO)
# Google Colab Code
# Includes your USN: 1BM23CS239
# ------------------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt

USN = "1BM23CS239"
print("Running GWO-Based Image Outline Deepening")
print("USN:", USN)
print("--------------------------------------------")

# ------------------------------------------------------------
# Upload an image
# ------------------------------------------------------------
from google.colab import files
uploaded = files.upload()

# Load the first uploaded image
img_path = list(uploaded.keys())[0]
img = cv2.imread(img_path, 0)  # grayscale
img = cv2.resize(img, (400, 400))  # normalize size
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()

# ------------------------------------------------------------
# Fitness Function (edge strength)
# ------------------------------------------------------------
def fitness(threshold):
    threshold = int(threshold)
    edges = cv2.Canny(img, threshold, threshold * 2)
    return -np.sum(edges)   # maximize edges → minimize negative edges


# ------------------------------------------------------------
# Grey Wolf Optimization (GWO)
# ------------------------------------------------------------
def GWO(max_iter=50, wolves=15, lb=1, ub=255):
    alpha, beta, delta = None, None, None
    a_values = []
    
    population = np.random.uniform(lb, ub, wolves)
    fitness_values = np.array([fitness(x) for x in population])

    for t in range(max_iter):
        # Sort by fitness (ascending, because lower = better)
        sorted_idx = np.argsort(fitness_values)
        alpha = population[sorted_idx[0]]
        beta  = population[sorted_idx[1]]
        delta = population[sorted_idx[2]]

        a = 2 - (2 * t) / max_iter
        a_values.append(fitness_values[sorted_idx[0]])

        new_population = []
        for i in range(wolves):
            r1, r2 = np.random.random(), np.random.random()
            A1 = 2*a*r1 - a
            C1 = 2*r2

            D_alpha = abs(C1 * alpha - population[i])
            X1 = alpha - A1 * D_alpha

            r1, r2 = np.random.random(), np.random.random()
            A2 = 2*a*r1 - a
            C2 = 2*r2

            D_beta = abs(C2 * beta - population[i])
            X2 = beta - A2 * D_beta

            r1, r2 = np.random.random(), np.random.random()
            A3 = 2*a*r1 - a
            C3 = 2*r2

            D_delta = abs(C3 * delta - population[i])
            X3 = delta - A3 * D_delta

            newX = (X1 + X2 + X3) / 3
            newX = np.clip(newX, lb, ub)
            new_population.append(newX)

        population = np.array(new_population)
        fitness_values = np.array([fitness(x) for x in population])

    best_threshold = alpha
    return int(best_threshold), a_values


# ------------------------------------------------------------
# Run GWO
# ------------------------------------------------------------
best_threshold, history = GWO()

print("Best Threshold (GWO optimized):", best_threshold)

# Apply edge enhancement using best threshold
edges = cv2.Canny(img, best_threshold, best_threshold * 2)

# Deepening outline: multiply edges with image
outline_deepened = cv2.addWeighted(img, 1, edges, 1, 0)

# ------------------------------------------------------------
# Plot results
# ------------------------------------------------------------

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(edges, cmap='gray')
plt.title(f"Edges (Threshold = {best_threshold})")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(outline_deepened, cmap='gray')
plt.title("Outline Deepened Image")
plt.axis('off')

plt.show()

# Convergence graph
plt.plot(history)
plt.title("GWO Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()

print("\nProcessing Completed Successfully for USN:", USN)
