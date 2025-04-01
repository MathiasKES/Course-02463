import matplotlib.pyplot as plt
import networkx as nx

import numpy as np

# Create a directed graph
G = nx.DiGraph()

# Add nodes for the variables
nodes = list(set(["F", "J", "H", "K", "M", "Q", "B", "Z", "Q", "S", "U", "W", "Y", "O", "P"]))

# Add edges based on causal relationships
edges = [
    ("F", "J"), ("H", "J"),   # Clause 1
    ("K", "M"), ("M", "Q"), ("Q", "B"),  # Clause 2
    ("Q", "Z"), ("S", "Z"), ("U", "Z"), ("W", "Z"), ("Y", "Z"),  # Clause 3
    ("H", "Y"),  # Clause 4
    ("J", "O"), ("J", "U"), ("O", "P"), ("U", "P"),  # Clause 5
    ("O", "Q")   # Clause 6
]

G.add_nodes_from(nodes)
G.add_edges_from(edges)

seed = np.random.randint(1000000000000)
pos = nx.spring_layout(G, seed)  # Layout for readability
print(seed)


# Draw the graph
plt.figure(figsize=(10, 10))
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1000, edge_color="black", font_size=8, arrows=True)
plt.title("Causal Diagram of the Ideal Gas Law", fontsize=10)
plt.show()
