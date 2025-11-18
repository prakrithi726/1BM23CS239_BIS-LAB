# Particle Swarm Optimization for Image Processing
#  USN Display: 1BM23CS239

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

USN = "1BM23CS239"   # <<< YOUR USN HERE


# -----------------------------------
# 1. Upload Image
# -----------------------------------
print("Upload an image (any format).")
uploaded = files.upload()

for fn in uploaded.keys():
    input_image_path = fn

# Read image
img = cv2.imread(input_image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# -----------------------------------
# 2. PSO Parameters
# -----------------------------------
num_particles = 30
max_iter = 50
w = 0.7     # inertia
c1 = 1.5    # cognitive
c2 = 1.5    # social


# -----------------------------------
# 3. Fitness Function (Otsu-inspired)
# -----------------------------------
def fitness(threshold):
    threshold = int(threshold)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Otsu-like variance metric (higher is better)
    pixels = gray.flatten()

    class0 = pixels[pixels <= threshold]
    class1 = pixels[pixels > threshold]

    if len(class0) == 0 or len(class1) == 0:
        return 0

    w0 = len(class0) / len(pixels)
    w1 = len(class1) / len(pixels)

    m0 = np.mean(class0)
    m1 = np.mean(class1)

    return w0 * w1 * ((m0 - m1)**2)


# -----------------------------------
# 4. Initialize PSO
# -----------------------------------
particles = np.random.uniform(0, 255, num_particles)      # thresholds
velocity = np.zeros(num_particles)

pbest = particles.copy()
pbest_fitness = np.array([fitness(t) for t in particles])

gbest = pbest[np.argmax(pbest_fitness)]
gbest_fitness = max(pbest_fitness)


# -----------------------------------
# 5. PSO Loop
# -----------------------------------
fitness_history = []

for i in range(max_iter):
    for j in range(num_particles):

        # Update velocity
        r1, r2 = np.random.rand(), np.random.rand()
        velocity[j] = (w * velocity[j] +
                       c1 * r1 * (pbest[j] - particles[j]) +
                       c2 * r2 * (gbest - particles[j]))

        # Update particle
        particles[j] += velocity[j]

        # Clamp threshold
        particles[j] = np.clip(particles[j], 0, 255)

        # Fitness
        f = fitness(particles[j])

        # Update personal best
        if f > pbest_fitness[j]:
            pbest[j] = particles[j]
            pbest_fitness[j] = f

            # Update global best
            if f > gbest_fitness:
                gbest = particles[j]
                gbest_fitness = f

    fitness_history.append(gbest_fitness)
    print(f"Iteration {i+1}/{max_iter}, Best Threshold = {int(gbest)}, Fitness = {gbest_fitness:.4f}")

best_threshold = int(gbest)
print("\nPSO Complete!")
print("Best Threshold:", best_threshold)
print("USN:", USN)

# -----------------------------------
# 6. Final Binarized Output
# -----------------------------------
_, binary_final = cv2.threshold(gray, best_threshold, 255, cv2.THRESH_BINARY)


# -----------------------------------
# 7. Display Results
# -----------------------------------
plt.figure(figsize=(15,6))

plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.title(f"Original Image\nUSN: {USN}")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(binary_final, cmap='gray')
plt.title(f"Binarized Image (PSO Threshold={best_threshold})")
plt.axis("off")

plt.show()


# -----------------------------------
# 8. Plot PSO Convergence
# -----------------------------------
plt.figure(figsize=(8,4))
plt.plot(fitness_history, marker='o')
plt.title(f"PSO Fitness Convergence (USN: {USN})")
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.grid(True)
plt.show()
