import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add nodes
nodes = ["Job Type (J)", "Health (H)", "Retirement Age (R)", "Death (D)", "Age (A)", "Economy (E)"]
G.add_nodes_from(nodes)

# Add edges based on causal relationships
edges = [("Job Type (J)", "Health (H)"),
         ("Job Type (J)", "Retirement Age (R)"),
         ("Age (A)", "Health (H)"),
         ("Age (A)", "Death (D)"),
         ("Health (H)", "Death (D)"),
         ("Economy (E)", "Health (H)"),
         ("Economy (E)", "Retirement Age (R)"),
         ("Retirement Age (R)", "Death (D)")]
G.add_edges_from(edges)

# Draw the graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)  # Layout for readability
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="black", node_size=3000, font_size=5, font_weight="bold")
plt.title("Causal Diagram of Retirement and Mortality")
plt.show()
