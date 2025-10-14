import osmnx as ox
import matplotlib.pyplot as plt

print("Baixando grafo de Brasília (região administrativa oficial)...")

G = ox.graph_from_place("Distrito Federal, Brazil", network_type="drive")

ox.save_graphml(G, "brasilia.graphml")
print("Grafo salvo como 'brasilia.graphml'.")

fig, ax = ox.plot_graph(G, node_size=0, edge_color="gray", figsize=(10, 10))
plt.show()
