# Ant Colony Optimization for Packet Routing
# Includes graph visualization + USN output
# USN: 1BM23CS239

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

USN = "1BM23CS239"   # <<< YOUR USN HERE

# ---------------------------------------
# 1. Create Network Graph
# ---------------------------------------
G = nx.Graph()

edges = [
    ("A","B",4), ("A","C",3),
    ("B","D",2), ("C","D",4),
    ("C","E",2), ("D","E",3),
    ("D","F",2), ("E","F",3)
]

# Add weighted edges
for u,v,w in edges:
    G.add_edge(u, v, weight=w)

pos = nx.spring_layout(G, seed=42)   # Node layout


# ---------------------------------------
# 2. ACO Parameters
# ---------------------------------------
num_ants = 15
num_iterations = 40
alpha = 1        # pheromone importance
beta = 2         # heuristic importance (1/distance)
evaporation = 0.5
pheromone_constant = 100

nodes = list(G.nodes)
pheromone = np.ones((len(nodes), len(nodes)))

node_index = {node:i for i,node in enumerate(nodes)}
index_node = {i:node for node,i in node_index.items()}


# ---------------------------------------
# 3. Choose Source and Destination
# ---------------------------------------
start = "A"
end = "F"

print("Source:", start)
print("Destination:", end)
print("USN:", USN)


# ---------------------------------------
# 4. Helper: Path Distance
# ---------------------------------------
def path_distance(path):
    dist = 0
    for i in range(len(path)-1):
        dist += G[path[i]][path[i+1]]['weight']
    return dist


# ---------------------------------------
# 5. ACO Algorithm
# ---------------------------------------
best_path = None
best_cost = float("inf")
cost_history = []

for iteration in range(num_iterations):
    all_paths = []
    all_costs = []

    # -------------------------------------------------
    # Each ant constructs a path
    # -------------------------------------------------
    for ant in range(num_ants):
        path = [start]
        current = start

        visited = set([start])

        while current != end:
            neighbors = list(G.neighbors(current))

            # Probabilities based on pheromone & heuristic
            probs = []
            for nxt in neighbors:
                if nxt not in visited:
                    i = node_index[current]
                    j = node_index[nxt]
                    tau = pheromone[i][j]
                    eta = 1.0 / G[current][nxt]['weight']
                    probs.append((nxt, (tau**alpha) * (eta**beta)))

            if len(probs) == 0:
                break

            nodes_list, prob_values = zip(*probs)
            prob_values = np.array(prob_values)
            prob_values = prob_values / prob_values.sum()

            next_node = np.random.choice(nodes_list, p=prob_values)
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        # Evaluate path
        if path[-1] == end:
            cost = path_distance(path)
            all_paths.append(path)
            all_costs.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_path = path

    # -------------------------------------------------
    # Pheromone Update
    # -------------------------------------------------
    pheromone = (1 - evaporation) * pheromone  # evaporation

    for i, path in enumerate(all_paths):
        cost = all_costs[i]
        deposit = pheromone_constant / cost
        for k in range(len(path) - 1):
            a = node_index[path[k]]
            b = node_index[path[k + 1]]
            pheromone[a][b] += deposit
            pheromone[b][a] += deposit

    cost_history.append(best_cost)
    print(f"Iteration {iteration+1}/{num_iterations}, Best Cost = {best_cost}, Best Path = {best_path}")


# ---------------------------------------
# 6. Final Best Route
# ---------------------------------------
print("\nBest Route Found:", best_path)
print("Best Path Cost:", best_cost)
print("USN:", USN)

# ---------------------------------------
# 7. Draw Network + Best Route
# ---------------------------------------
plt.figure(figsize=(10,6))

# Draw full network
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1400, font_size=12)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

# Highlight best path
best_edges = [(best_path[i], best_path[i+1]) for i in range(len(best_path)-1)]
nx.draw_networkx_edges(G, pos, edgelist=best_edges, width=4, edge_color='red')

plt.title(f"ACO Packet Routing Graph\nBest Path: {best_path} | USN: {USN}")
plt.show()

# ---------------------------------------
# 8. Plot Cost Convergence
# ---------------------------------------
plt.figure(figsize=(8,4))
plt.plot(cost_history, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Path Cost")
plt.grid(True)
plt.title(f"ACO Cost Convergence (USN: {USN})")
plt.show()

output:
Source: A
Destination: F
USN: 1BM23CS239
Iteration 1/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 2/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 3/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 4/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 5/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 6/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 7/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 8/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 9/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 10/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 11/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 12/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 13/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 14/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 15/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 16/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 17/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 18/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 19/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 20/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 21/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 22/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 23/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 24/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 25/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 26/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 27/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 28/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 29/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 30/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 31/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 32/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 33/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 34/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 35/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 36/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 37/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 38/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 39/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Iteration 40/40, Best Cost = 8, Best Path = ['A', np.str_('B'), np.str_('D'), np.str_('F')]

Best Route Found: ['A', np.str_('B'), np.str_('D'), np.str_('F')]
Best Path Cost: 8
USN: 1BM23CS239
