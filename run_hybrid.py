"""
run_hybrid.py - Versão híbrida que usa OSM cached ou Haversine

Tenta carregar dados OSM salvos localmente primeiro,
se não conseguir, usa distâncias Haversine.
"""

import math
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

# OSMnx 2.x configuration
ox.settings.use_cache = True
ox.settings.log_console = False

# -----------------------------
# Dados e parâmetros
# -----------------------------
CAPACIDADE = 1800  # kg
VEL_KMH = 50.0  # velocidade média (km/h)
TEMPO_MAX_DIA_MIN = 6 * 60  # 6 horas em minutos

# Clientes: ID 0 é depósito
clientes = {
    0: {"nome": "Depósito (Carrefour STN)", "lat": -15.7366, "lon": -47.90732, "demanda": 0, "descarga": 515},
    1: {"nome": "CLS 307", "lat": -15.8122664, "lon": -47.9013959, "demanda": 160, "descarga": 50},
    2: {"nome": "CLS 114", "lat": -15.8268977, "lon": -47.9191361, "demanda": 170, "descarga": 60},
    3: {"nome": "CLN 110", "lat": -15.7743127, "lon": -47.88647, "demanda": 22, "descarga": 70},
    4: {"nome": "SOF (Água Mineral)", "lat": -15.738056, "lon": -47.926667, "demanda": 300, "descarga": 85},
    5: {"nome": "SHIS QI 17 (Lago Sul)", "lat": -15.845, "lon": -47.862, "demanda": 250, "descarga": 45},
    6: {"nome": "CLSW 103", "lat": -15.8010635, "lon": -47.9248713, "demanda": 90, "descarga": 65},
    7: {"nome": "Varjão (entrada)", "lat": -15.70972, "lon": -47.87889, "demanda": 130, "descarga": 55},
    8: {"nome": "Águas Claras (shopping)", "lat": -15.84028, "lon": -48.02778, "demanda": 350, "descarga": 40},
    9: {"nome": "Taguatinga Pistão Sul", "lat": -15.851861, "lon": -48.041972, "demanda": 900, "descarga": 45},
}

ids = sorted(clientes.keys())

# -----------------------------
# Funções utilitárias
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def travel_time_minutes_km(distance_km, vel_kmh=VEL_KMH):
    return (distance_km / vel_kmh) * 60.0

def route_service_time_minutes(route):
    t = 0.0
    for node_id in route:
        t += clientes[node_id]["descarga"] if node_id != 0 else 0.0
    return t

# -----------------------------
# Carregar grafo OSM (se disponível)
# -----------------------------
def load_osm_graph():
    graph_file = 'brasilia_graph.pkl'
    if os.path.exists(graph_file):
        print(f"Carregando grafo OSM salvo de {graph_file}...")
        try:
            with open(graph_file, 'rb') as f:
                G = pickle.load(f)
            print("✓ Grafo OSM carregado com sucesso!")
            return G
        except Exception as e:
            print(f"Erro ao carregar grafo: {e}")
            return None
    else:
        print("Arquivo de grafo OSM não encontrado.")
        print("Execute: python download_osm.py para criar o cache.")
        return None

# Tentar carregar grafo OSM
G = load_osm_graph()

# Mapear coordenadas para nós do grafo (se disponível)
nearest_node = None
if G is not None:
    print("Mapeando clientes para nós OSM...")
    nearest_node = {}
    for i in ids:
        lat, lon = clientes[i]["lat"], clientes[i]["lon"]
        try:
            nearest_node[i] = ox.distance.nearest_nodes(G, lon, lat)
        except Exception as e:
            print(f"Erro ao mapear cliente {i}: {e}")
            nearest_node[i] = None

# -----------------------------
# Calcular matriz de distâncias
# -----------------------------
print("Calculando matriz de distâncias...")
n = len(ids)
dist_km = np.zeros((n, n))
time_min = np.zeros((n, n))
id_to_index = {node_id: idx for idx, node_id in enumerate(ids)}

using_osm = G is not None and nearest_node is not None
print(f"Método: {'OSM + Shortest Path' if using_osm else 'Haversine'}")

for i_idx, i in enumerate(ids):
    for j_idx, j in enumerate(ids):
        if i == j:
            dist_km[i_idx, j_idx] = 0.0
            time_min[i_idx, j_idx] = 0.0
            continue
        
        # Tentar usar OSM se disponível
        if using_osm and nearest_node.get(i) is not None and nearest_node.get(j) is not None:
            u = nearest_node[i]
            v = nearest_node[j]
            try:
                length_m = nx.shortest_path_length(G, u, v, weight="length")
                km = length_m / 1000.0
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # fallback to haversine
                km = haversine_km(clientes[i]["lat"], clientes[i]["lon"], clientes[j]["lat"], clientes[j]["lon"])
        else:
            # usar haversine diretamente
            km = haversine_km(clientes[i]["lat"], clientes[i]["lon"], clientes[j]["lat"], clientes[j]["lon"])
        
        dist_km[i_idx, j_idx] = km
        time_min[i_idx, j_idx] = travel_time_minutes_km(km)

print("Matriz de distâncias (km):")
print(np.round(dist_km, 3))

# -----------------------------
# [Resto do código igual ao run_fast.py]
# -----------------------------
def route_distance_and_time(route):
    total_km = 0.0
    total_time_min = 0.0
    for a, b in zip(route[:-1], route[1:]):
        i = id_to_index[a]
        j = id_to_index[b]
        total_km += dist_km[i, j]
        total_time_min += time_min[i, j]
    total_time_min += route_service_time_minutes(route)
    return total_km, total_time_min

def route_load(route):
    return sum(clientes[node]["demanda"] for node in route if node != 0)

# [Incluir todas as heurísticas do run_fast.py - código omitido para brevidade]
# ... (sweep_routes, nearest_neighbor_routes, clarke_wright_routes) ...

print("\nPara executar com dados OSM mais precisos:")
print("1. Execute: python download_osm.py")
print("2. Execute: python run_hybrid.py")