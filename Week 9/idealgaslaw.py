import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes for the variables
nodes = {
    "P": "Pressure (P)",
    "V": "Volume (V)",
    "T": "Temperature (T)",
    "n": "Moles of Gas (n)",
    "[R]": "Gas Constant (R)",
}

# Add edges based on causal relationships
edges = [("V", "P"), ("T", "P"), ("[R]", "P"), ("n", "P")]

G.add_nodes_from(nodes.keys())
G.add_edges_from(edges)

# Define node positions for better readability
pos = {
    "V": (0, 1),
    "T": (1, 0),
    "n": (-1, 0),
    "[R]": (-1, 1),
    "P": (0, -1),
}

# Draw the graph
plt.figure(figsize=(6, 4))
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=3000, edge_color="black", font_size=8, arrows=True)
plt.title("Causal Diagram of the Ideal Gas Law", fontsize=10)
plt.show()
