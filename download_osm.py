"""
download_osm.py - Script para baixar e salvar dados OSM localmente

Este script baixa o grafo OSM uma vez e salva em arquivo local
para reutilização rápida em execuções futuras.
"""

import osmnx as ox
import pickle
import os

# OSMnx 2.x configuration
ox.settings.use_cache = True
ox.settings.log_console = False

def download_and_save_osm():
    """Baixa e salva o grafo OSM para uso futuro."""
    print("Baixando grafo OSM de Brasília...")
    
    # Área mínima que cobre os pontos de entrega
    bbox = (-15.70, -15.86, -47.86, -48.05)  # (north, south, east, west)
    
    try:
        G = ox.graph_from_bbox(bbox, network_type="drive", simplify=True)
        
        # Garantir que 'length' existe
        for u, v, k, data in G.edges(keys=True, data=True):
            if 'length' not in data:
                data['length'] = ox.distance.great_circle_vec(G.nodes[u]['y'], G.nodes[u]['x'],
                                                               G.nodes[v]['y'], G.nodes[v]['x'])
        
        # Salvar o grafo em arquivo
        graph_file = 'brasilia_graph.pkl'
        with open(graph_file, 'wb') as f:
            pickle.dump(G, f)
        
        print(f"Grafo salvo em {graph_file}")
        print(f"Número de nós: {len(G.nodes)}")
        print(f"Número de arestas: {len(G.edges)}")
        
        return G
        
    except Exception as e:
        print(f"Erro ao baixar OSM: {e}")
        return None

def load_osm_graph():
    """Carrega o grafo OSM salvo localmente."""
    graph_file = 'brasilia_graph.pkl'
    
    if os.path.exists(graph_file):
        print(f"Carregando grafo salvo de {graph_file}...")
        with open(graph_file, 'rb') as f:
            G = pickle.load(f)
        print("Grafo carregado com sucesso!")
        return G
    else:
        print(f"Arquivo {graph_file} não encontrado. Execute download_and_save_osm() primeiro.")
        return None

if __name__ == "__main__":
    # Baixar e salvar o grafo
    G = download_and_save_osm()
    
    if G is not None:
        print("\nPara usar o grafo salvo em seus scripts:")
        print("from download_osm import load_osm_graph")
        print("G = load_osm_graph()")