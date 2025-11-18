# --------------------------------------------
# Optimization in Power Consumption using
# Cuckoo Search Algorithm (CSA)

# Includes graph and prints USN: 1BM23CS239
# --------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import math # Import the math module

# Your USN
USN = "1BM23CS239"
print("Running Cuckoo Search Optimization")
print("USN:", USN)
print("------------------------------------------")


# -------------------------------------------------------------
# Define the Power Consumption Model (example)
# P(x) = a*x^2 + b*x + c  (x = operating load percentage)
# -------------------------------------------------------------
def power_consumption(x):
    a, b, c = 0.02, -1.5, 200   # sample model
    return a*(x**2) + b*x + c


# -------------------------------------------------------------
# Cuckoo Search Algorithm Implementation
# -------------------------------------------------------------
def cuckoo_search(n=20, iterations=200, pa=0.25, lb=0, ub=100):
    # Generate initial population
    population = np.random.uniform(lb, ub, n)
    fitness = np.array([power_consumption(x) for x in population])

    best_index = np.argmin(fitness)
    best = population[best_index]
    best_fitness = fitness[best_index]

    history = []

    for t in range(iterations):
        # Levy flight
        beta = 1.5
        sigma = ( math.gamma(1+beta) * np.sin(np.pi*beta/2) \
                 / ( math.gamma((1+beta)/2) * beta*2**((beta-1)/2) ) )**(1/beta)

        levy = np.random.normal(0, sigma, n) / np.abs(np.random.normal(0, 1, n))**(1/beta)

        new_population = population + levy

        # Bounds check
        new_population = np.clip(new_population, lb, ub)

        new_fitness = np.array([power_consumption(x) for x in new_population])

        # Greedy replacement
        for i in range(n):
            if new_fitness[i] < fitness[i]:
                population[i] = new_population[i]
                fitness[i] = new_fitness[i]

        # Abandon nests
        abandon = np.random.rand(n) < pa
        random_new = np.random.uniform(lb, ub, sum(abandon))
        population[abandon] = random_new
        fitness[abandon] = np.array([power_consumption(x) for x in random_new])

        # Track best
        best_index = np.argmin(fitness)
        best = population[best_index]
        best_fitness = fitness[best_index]

        history.append(best_fitness)

    return best, best_fitness, history


# -------------------------------------------------------------
# Run the Optimization
# -------------------------------------------------------------
best_x, best_cost, history = cuckoo_search()

print("Optimal Operating Load (x):", best_x)
print("Minimum Power Consumption Cost:", best_cost)


# -------------------------------------------------------------
# Plotting Results
# -------------------------------------------------------------

# 1. Fitness graph
plt.figure(figsize=(8,5))
plt.plot(history)
plt.title("Cuckoo Search Optimization – Power Consumption")
plt.xlabel("Iteration")
plt.ylabel("Best Power Cost")
plt.grid(True)
plt.show()

# 2. Plot original power function & optimized point
x_vals = np.linspace(0,100,400)
y_vals = power_consumption(x_vals)

plt.figure(figsize=(8,5))
plt.plot(x_vals, y_vals, label="Power Consumption Curve")
plt.scatter([best_x], [best_cost], color='red', s=80, label="Optimized Point")
plt.title("Power Consumption vs Load (%)")
plt.xlabel("Load (%)")
plt.ylabel("Power Cost")
plt.legend()
plt.grid(True)
plt.show()

print("\nOptimization Completed Successfully for USN:", USN)
