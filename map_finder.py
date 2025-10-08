import osmnx as ox
import matplotlib.pyplot as plt


# -----------------------------
# Bounding Box definido (Brasília área central)
# -----------------------------
# Coordenadas: (south, north, west, east)
south, north = -15.938882, -15.699731
west, east = -48.208120, -47.812497

# -----------------------------
# Baixar grafo de ruas para carro (drive)
# -----------------------------
print("Baixando grafo de Brasília (área do bounding box)...")
G = ox.graph_from_bbox([north, south, east, west], network_type="drive")

# -----------------------------
# Salvar grafo em arquivo GraphML (opcional)
# -----------------------------
ox.save_graphml(G, "brasilia_bbox.graphml")
print("Grafo salvo como 'brasilia_bbox.graphml'.")

# -----------------------------
# Plotar grafo
# -----------------------------
fig, ax = ox.plot_graph(G, node_size=0, edge_color="gray", figsize=(10,10))
plt.show()
