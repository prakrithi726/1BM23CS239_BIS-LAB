# Robotic Process + Genetic Algorithm
# Paste into Google Colab and run.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches
from IPython.display import HTML
import random

# -----------------------------
# Robot & environment settings
# -----------------------------
DT = 0.2            # timestep for each command (seconds)
N_COMMANDS = 30     # number of sequential commands in a chromosome
WORLD_BOUNDS = (-5, 5, -5, 5)  # x_min, x_max, y_min, y_max

start = np.array([-4.0, -4.0, 0.0])   # x, y, theta
goal = np.array([4.0, 4.0])           # goal position (x, y)
goal_radius = 0.2                     # considered reached if within this radius

# command bounds
V_MIN, V_MAX = -1.0, 1.0    # linear velocity bounds (m/s)
W_MIN, W_MAX = -2.0, 2.0    # angular velocity bounds (rad/s)

# -----------------------------
# GA hyperparameters
# -----------------------------
POP_SIZE = 150
GENERATIONS = 120
TOURNAMENT_SIZE = 3
ELITISM = 2
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.12
MUTATION_STD_V = 0.2
MUTATION_STD_W = 0.4
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# -----------------------------
# Chromosome representation:
# sequence of (v, w) for N_COMMANDS
# -----------------------------
def create_random_chromosome():
    # shape (N_COMMANDS, 2)
    v = np.random.uniform(V_MIN, V_MAX, size=N_COMMANDS)
    w = np.random.uniform(W_MIN, W_MAX, size=N_COMMANDS)
    return np.vstack([v, w]).T

# -----------------------------
# Robot simulation (kinematic)
# -----------------------------
def simulate(chromosome, start_state=start, dt=DT, record_poses=False):
    """
    Simulate the robot executing the sequence of (v, w) commands.
    Returns final state and optionally the full pose trajectory.
    state: [x, y, theta]
    """
    x, y, th = start_state.copy()
    poses = [(x, y, th)]
    for (v, w) in chromosome:
        # simple forward Euler integration of unicycle model
        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        th += w * dt
        # normalize theta to [-pi, pi]
        th = (th + np.pi) % (2*np.pi) - np.pi
        poses.append((x, y, th))
    if record_poses:
        return np.array(poses)
    else:
        return np.array([x, y, th])

# -----------------------------
# Fitness function
# -----------------------------
def fitness(chromosome, start_state=start):
    """
    Higher fitness is better. We use negative final distance to goal plus bonuses.
    We also reward reaching the goal early (shorter time).
    """
    poses = simulate(chromosome, start_state=start_state, record_poses=True)
    final_pos = poses[-1, :2]
    dist = np.linalg.norm(final_pos - goal)
    # base fitness: inverse of distance
    fitness_value = -dist
    # bonus if end close to goal
    if dist <= goal_radius:
        # reward more if reached earlier (index is time step)
        # find earliest time within goal radius
        dists = np.linalg.norm(poses[:, :2] - goal.reshape(1,2), axis=1)
        idx = np.argwhere(dists <= goal_radius)
        if idx.size > 0:
            t_reached = idx[0, 0]
            # bigger reward for earlier reach
            fitness_value += 3.0 + 1.0 * (N_COMMANDS - t_reached) / N_COMMANDS
    # penalty for leaving world bounds
    x, y = poses[-1,0], poses[-1,1]
    x_min, x_max, y_min, y_max = WORLD_BOUNDS
    if not (x_min <= x <= x_max and y_min <= y <= y_max):
        fitness_value -= 5.0
    return fitness_value

# -----------------------------
# GA operators
# -----------------------------
def tournament_selection(pop, fitnesses, k=TOURNAMENT_SIZE):
    idxs = np.random.randint(0, len(pop), k)
    best_idx = idxs[0]
    for i in idxs:
        if fitnesses[i] > fitnesses[best_idx]:
            best_idx = i
    return pop[best_idx].copy()

def uniform_crossover(a, b):
    """Uniform crossover on each gene (v,w pair)"""
    child = a.copy()
    mask = np.random.rand(N_COMMANDS) < 0.5
    child[mask] = b[mask]
    return child

def mutate(chromosome, rate=MUTATION_RATE):
    for i in range(len(chromosome)):
        if np.random.rand() < rate:
            # mutate both v and w with gaussian noise and clamp
            chromosome[i,0] += np.random.normal(0, MUTATION_STD_V)
            chromosome[i,1] += np.random.normal(0, MUTATION_STD_W)
            chromosome[i,0] = np.clip(chromosome[i,0], V_MIN, V_MAX)
            chromosome[i,1] = np.clip(chromosome[i,1], W_MIN, W_MAX)
    return chromosome

# -----------------------------
# Initialize population
# -----------------------------
population = [create_random_chromosome() for _ in range(POP_SIZE)]
best_history = []
avg_history = []

# -----------------------------
# Evolution loop
# -----------------------------
for gen in range(GENERATIONS):
    fitnesses = np.array([fitness(ind) for ind in population])
    sorted_idx = np.argsort(-fitnesses)  # descending
    best_fitness = fitnesses[sorted_idx[0]]
    avg_fitness = fitnesses.mean()
    best_history.append(best_fitness)
    avg_history.append(avg_fitness)

    # print progress every N gens
    if gen % 10 == 0 or gen == GENERATIONS-1:
        print(f"Gen {gen:3d} | Best fitness: {best_fitness:.4f} | Avg: {avg_fitness:.4f}")

    # create next population with elitism
    next_pop = []
    # Elitism: copy top ELITISM individuals
    for i in range(ELITISM):
        next_pop.append(population[sorted_idx[i]].copy())

    # fill the rest
    while len(next_pop) < POP_SIZE:
        parent1 = tournament_selection(population, fitnesses)
        parent2 = tournament_selection(population, fitnesses)
        # crossover
        if np.random.rand() < CROSSOVER_RATE:
            child = uniform_crossover(parent1, parent2)
        else:
            child = parent1.copy()
        # mutation
        child = mutate(child)
        next_pop.append(child)
    population = next_pop

# After evolution: get best individual
fitnesses = np.array([fitness(ind) for ind in population])
best_idx = np.argmax(fitnesses)
best_ind = population[best_idx]
print("\nFinished. Best final fitness:", fitnesses[best_idx])

# -----------------------------
# Visualize results
# -----------------------------
best_poses = simulate(best_ind, start_state=start, record_poses=True)

# Plot fitness evolution
plt.figure(figsize=(9,4))
plt.plot(best_history, label="best")
plt.plot(avg_history, label="avg")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("GA fitness over generations")
plt.legend()
plt.grid(True)
plt.show()

# Static plot of the best trajectory
fig, ax = plt.subplots(figsize=(6,6))
x_min, x_max, y_min, y_max = WORLD_BOUNDS
ax.set_xlim(x_min-0.5, x_max+0.5)
ax.set_ylim(y_min-0.5, y_max+0.5)
ax.set_aspect('equal')

# Draw start and goal
ax.plot(start[0], start[1], 'go', label='start')
ax.plot(goal[0], goal[1], 'r*', markersize=14, label='goal')
ax.add_patch(plt.Circle((goal[0], goal[1]), goal_radius, color='red', alpha=0.15))

# Draw trajectory
traj_xy = best_poses[:, :2]
ax.plot(traj_xy[:,0], traj_xy[:,1], '-o', markersize=3, linewidth=1, label='best trajectory')

ax.set_title('Best trajectory (final)')
ax.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Animation (inline in colab)
# -----------------------------
fig2, ax2 = plt.subplots(figsize=(6,6))
ax2.set_xlim(x_min-0.5, x_max+0.5)
ax2.set_ylim(y_min-0.5, y_max+0.5)
ax2.set_aspect('equal')

ax2.plot(start[0], start[1], 'go', label='start')
ax2.plot(goal[0], goal[1], 'r*', markersize=14, label='goal')
ax2.add_patch(plt.Circle((goal[0], goal[1]), goal_radius, color='red', alpha=0.15))

line, = ax2.plot([], [], '-o', markersize=5)
robot_patch = patches.Circle((start[0], start[1]), 0.12, fc='blue')
ax2.add_patch(robot_patch)

def init_anim():
    line.set_data([], [])
    robot_patch.center = (start[0], start[1])
    return line, robot_patch

def animate(i):
    if i >= len(best_poses):
        i = len(best_poses)-1
    xs = best_poses[:i+1, 0]
    ys = best_poses[:i+1, 1]
    line.set_data(xs, ys)
    robot_patch.center = (best_poses[i,0], best_poses[i,1])
    return line, robot_patch

anim = animation.FuncAnimation(fig2, animate, init_func=init_anim,
                               frames=len(best_poses), interval=120, blit=True)

# Display animation in notebook
HTML(anim.to_jshtml())
