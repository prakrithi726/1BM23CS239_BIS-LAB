# ------------------------------------------------------------
# IMAGE PROCESSING USING GA + GEP (Gene Expression Programming)

# Includes USN: 1BM23CS239
# ------------------------------------------------------------

import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab import files
import random
import math

USN = "1BM23CS239"
print("Running Image Processing using GA + GEP")
print("USN:", USN)
print("-------------------------------------------------")


# ------------------------------------------------------------
# Upload Image
# ------------------------------------------------------------
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

img = cv2.imread(image_path, 0)
img = cv2.resize(img, (400, 400))

plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()


# ------------------------------------------------------------
# -------------- GENETIC ALGORITHM (GA) PART -----------------
# ------------------------------------------------------------

def fitness(th):
    # variance-based fitness for thresholding
    ret, thresh = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
    return np.var(thresh)

def GA_optimize(pop_size=20, generations=40):
    population = np.random.randint(0, 255, pop_size)

    for gen in range(generations):
        scores = np.array([fitness(p) for p in population])
        parents = population[np.argsort(scores)][-10:]

        children = []
        for _ in range(pop_size):
            p1, p2 = random.sample(list(parents), 2)
            child = (p1 + p2) // 2
            if random.random() < 0.1:
                child += random.randint(-15, 15)
            child = np.clip(child, 0, 255)
            children.append(child)

        population = np.array(children)

    best = population[np.argmax([fitness(p) for p in population])]
    return int(best)


best_threshold_ga = GA_optimize()
print("GA Optimal Threshold =", best_threshold_ga)

_, ga_img = cv2.threshold(img, best_threshold_ga, 255, cv2.THRESH_BINARY)


# ------------------------------------------------------------
# -------------- GENE EXPRESSION PROGRAMMING (GEP) -----------
# ------------------------------------------------------------

# Terminals: pixel intensity (x), constants
# Functions: +, -, *, /, sqrt, sin, cos

functions = ["+", "-", "*", "/", "sin", "cos", "sqrt"]
terminals = ["x", "50", "100", "150"]

def random_gene(length=10):
    return [random.choice(functions + terminals) for _ in range(length)]

def evaluate_gene(gene, x):
    stack = []
    for g in gene:
        if g in terminals:
            if g == "x":
                stack.append(x)
            else:
                stack.append(float(g))
        elif g in functions:
            if g in ["sin", "cos", "sqrt"]:
                if not stack:
                    stack.append(0)
                a = stack.pop()
                try:
                    if g == "sin": stack.append(math.sin(a))
                    if g == "cos": stack.append(math.cos(a))
                    if g == "sqrt": stack.append(math.sqrt(abs(a)))
                except:
                    stack.append(0)
            else:
                # Binary operators: "+", "-", "*", "/"
                # Ensure there are enough operands for binary operations
                val_b = stack.pop() if stack else 0
                val_a = stack.pop() if stack else 0
                try:
                    if g == "+": stack.append(val_a + val_b)
                    if g == "-": stack.append(val_a - val_b)
                    if g == "*": stack.append(val_a * val_b)
                    if g == "/":
                        if val_b == 0: stack.append(0)
                        else: stack.append(val_a / val_b)
                except:
                    stack.append(0)

    return stack[-1] if stack else 0


def gep_threshold(gene):
    val = evaluate_gene(gene, 128)
    return int(np.clip(val * 20, 0, 255))


def GEP_optimize(pop_size=20, generations=30, gene_length=10):
    population = [random_gene(gene_length) for _ in range(pop_size)]

    def fitness_gep(gene):
        th = gep_threshold(gene)
        ret, thresh = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
        return np.var(thresh)

    for gen in range(generations):
        scored = sorted(population, key=lambda g: fitness_gep(g), reverse=True)
        parents = scored[:10]

        children = []
        for _ in range(pop_size):
            p = random.choice(parents).copy()
            if random.random() < 0.3:
                idx = random.randint(0, len(p)-1)
                p[idx] = random.choice(terminals + functions)
            children.append(p)

        population = children

    best_gene = max(population, key=lambda g: fitness_gep(g))
    return best_gene


best_gene = GEP_optimize()
best_threshold_gep = gep_threshold(best_gene)

print("GEP Optimal Threshold =", best_threshold_gep)

_, gep_img = cv2.threshold(img, best_threshold_gep, 255, cv2.THRESH_BINARY)


# ------------------------------------------------------------
# Display Results
# ------------------------------------------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(ga_img, cmap='gray')
plt.title(f"GA Output (Th = {best_threshold_ga})")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(gep_img, cmap='gray')
plt.title(f"GEP Output (Th = {best_threshold_gep})")
plt.axis('off')

plt.show()

print("\nCompleted successfully for USN:", USN)
