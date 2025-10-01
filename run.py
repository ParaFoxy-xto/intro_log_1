"""
vrp_brasilia_osmnx.py

Script completo para:
 - baixar grafo viário de Brasília (OSMnx)
 - mapear clientes ao grafo
 - calcular matrizes de distância/tempo reais
 - gerar rotas com 4 heurísticas: Clarke & Wright, Nearest Neighbor,
   Farthest Neighbor e Sweep (por ângulo)
 - verificar capacidade (1800 kg) e tempo máximo por dia (6h)
 - produzir programação semanal (dias com rotas, sem exceder 6h/dia)

Autor: Gabriel (adaptado para seu dever)
"""

import math
import os
import sys
import time
from collections import defaultdict, deque
import itertools

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
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

# Lista de IDs (ordenada)
ids = sorted(clientes.keys())

# -----------------------------
# Funções utilitárias
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Distância em km pela fórmula haversine (usar só como fallback/checar)."""
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def travel_time_minutes_km(distance_km, vel_kmh=VEL_KMH):
    """Tempo de viagem em minutos para uma distância em km e velocidade média."""
    return (distance_km / vel_kmh) * 60.0

def route_service_time_minutes(route):
    """Tempo de serviço (descargas) somado nos nós internos (exclui depósito quando 0)."""
    t = 0.0
    for node_id in route:
        t += clientes[node_id]["descarga"] if node_id != 0 else 0.0
    return t

# -----------------------------
# Baixar grafo OSM de Brasília (drive) - versão simplificada
# -----------------------------
print("Baixando/obtendo grafo de Brasília (pode demorar alguns segundos)...")
# Usando coordenadas fornecidas pelo usuário
# Southwest: -15.938882, -48.208120, Northeast: -15.699731, -47.812497
bbox = (-15.699731, -15.938882, -47.812497, -48.208120)  # (north, south, east, west)
try:
    G = ox.graph_from_bbox(bbox, network_type="drive", simplify=True)
except Exception as e:
    print(f"Erro ao baixar grafo OSM: {e}")
    print("Usando distâncias haversine como fallback...")
    # Criar um grafo vazio para continuar com cálculos haversine
    G = None
# Garantir que 'length' existe (se o grafo foi carregado)
if G is not None:
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'length' not in data:
            data['length'] = ox.distance.great_circle_vec(G.nodes[u]['y'], G.nodes[u]['x'],
                                                           G.nodes[v]['y'], G.nodes[v]['x'])

# -----------------------------
# Mapear coordenadas para nós do grafo (se disponível)
# -----------------------------
if G is not None:
    print("Mapeando clientes para nós OSM (nearest nodes)...")
    coords = {i: (clientes[i]["lat"], clientes[i]["lon"]) for i in ids}
    # osmnx expects (lat, lon) but nearest_nodes takes (G, X, Y) where X=lon, Y=lat
    nearest_node = {}
    for i in ids:
        lat, lon = coords[i]
        try:
            nearest_node[i] = ox.distance.nearest_nodes(G, lon, lat)
        except Exception as e:
            print(f"Erro ao mapear cliente {i}: {e}")
            nearest_node[i] = None
else:
    print("Usando cálculos diretos de distância (sem mapeamento OSM)...")
    nearest_node = None

# -----------------------------
# Construir matriz de distâncias (km) e tempos (min)
# usando shortest_path_length com weight='length' (metros)
# -----------------------------
n = len(ids)
dist_km = np.zeros((n, n))
time_min = np.zeros((n, n))
id_to_index = {node_id: idx for idx, node_id in enumerate(ids)}
index_to_id = {idx: node_id for node_id, idx in id_to_index.items()}

print("Calculando matriz de distâncias reais (usando o grafo OSM ou haversine) — isso pode demorar um pouco...")
# Precompute pairwise shortest path lengths (meters)
# Use node IDs from nearest_node mapping

for i_idx, i in enumerate(ids):
    for j_idx, j in enumerate(ids):
        if i == j:
            dist_km[i_idx, j_idx] = 0.0
            time_min[i_idx, j_idx] = 0.0
            continue
        
        # Tentar usar OSM se disponível, senão usar haversine
        if G is not None and nearest_node is not None and nearest_node.get(i) is not None and nearest_node.get(j) is not None:
            u = nearest_node[i]
            v = nearest_node[j]
            try:
                length_m = nx.shortest_path_length(G, u, v, weight="length")
                km = length_m / 1000.0
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # fallback to haversine if no path found
                km = haversine_km(clientes[i]["lat"], clientes[i]["lon"], clientes[j]["lat"], clientes[j]["lon"])
        else:
            # usar haversine diretamente
            km = haversine_km(clientes[i]["lat"], clientes[i]["lon"], clientes[j]["lat"], clientes[j]["lon"])
        
        dist_km[i_idx, j_idx] = km
        time_min[i_idx, j_idx] = travel_time_minutes_km(km)

print("Matriz de distâncias (km):")
print(np.round(dist_km, 3))

# -----------------------------
# Funções para avaliar rotas
# -----------------------------
def route_distance_and_time(route):
    """
    route: lista de client IDs (ex: [0, 3, 5, 0])
    Retorna (distance_km, time_minutes_total) onde time inclui viagens + tempos de descarga.
    """
    total_km = 0.0
    total_time_min = 0.0
    for a, b in zip(route[:-1], route[1:]):
        i = id_to_index[a]
        j = id_to_index[b]
        total_km += dist_km[i, j]
        total_time_min += time_min[i, j]
    # adicionar tempo de descarga para nós não-depósito (se rota contém depósito como 0, ignore depositos)
    total_time_min += route_service_time_minutes(route)
    return total_km, total_time_min

def route_load(route):
    """Soma de demandas (kg) dos clientes no route (exclui depósito 0)."""
    return sum(clientes[node]["demanda"] for node in route if node != 0)

# -----------------------------
# Heurísticas
# -----------------------------
# 1) Sweep (varredura por ângulo)
def sweep_routes():
    depot = clientes[0]
    angles = []
    for i in ids:
        if i == 0: continue
        dx = clientes[i]["lon"] - depot["lon"]
        dy = clientes[i]["lat"] - depot["lat"]
        ang = math.atan2(dy, dx)
        angles.append((i, ang))
    angles.sort(key=lambda x: x[1])  # ordena por ângulo crescente
    routes = []
    current_route = [0]
    current_load = 0
    for i, ang in angles:
        demand = clientes[i]["demanda"]
        # Tentativa de inserir no route atual e checar se tempo e capacidade aceitam (simulando inserção no final)
        tentative_route = current_route + [i, 0]
        load_if = current_load + demand
        if load_if <= CAPACIDADE:
            # calcular tempo se inserirmos
            dist_km_val, time_min_val = route_distance_and_time(tentative_route)
            if time_min_val <= TEMPO_MAX_DIA_MIN:
                # aceita
                current_route = current_route + [i]
                current_load = load_if
            else:
                # fecha rota atual e inicia nova
                current_route.append(0)
                routes.append(current_route)
                current_route = [0, i]
                current_load = demand
        else:
            # fecha e inicia nova
            current_route.append(0)
            routes.append(current_route)
            current_route = [0, i]
            current_load = demand
    current_route.append(0)
    routes.append(current_route)
    return routes

# 2) Nearest Neighbor (Vizinho mais próximo) - constrói rotas greedy respeitando capacidade e tempo
def nearest_neighbor_routes(farthest=False):
    """Se farthest=False => nearest neighbor; farthest=True => pick farthest next (vizinho mais distante)."""
    unserved = set(i for i in ids if i != 0)
    routes = []
    while unserved:
        route = [0]
        load = 0
        cur = 0
        while True:
            candidates = list(unserved)
            if not candidates:
                break
            # escolher próximo (ou mais distante) ao nó corrente baseado em dist_km matrix
            cur_idx = id_to_index[cur]
            best = None
            best_val = None
            for c in candidates:
                val = dist_km[cur_idx, id_to_index[c]]
                if best is None:
                    best = c; best_val = val
                else:
                    if (not farthest and val < best_val) or (farthest and val > best_val):
                        best = c; best_val = val
            # testar se inserir best viola carga/tempo
            tentative_route = route + [best, 0]
            tentative_load = load + clientes[best]["demanda"]
            if tentative_load <= CAPACIDADE:
                dist_km_val, time_min_val = route_distance_and_time(tentative_route)
                if time_min_val <= TEMPO_MAX_DIA_MIN:
                    # aceitar
                    route.append(best)
                    load = tentative_load
                    unserved.remove(best)
                    cur = best
                    continue
            # se não aceitou, fechamos a rota atual
            break
        route.append(0)
        routes.append(route)
    return routes

# 3) Clarke & Wright Savings
def clarke_wright_routes():
    # Inicialmente cada cliente (não-depósito) em sua própria rota 0-i-0
    routes = {i: [0, i, 0] for i in ids if i != 0}
    route_loads = {i: clientes[i]["demanda"] for i in ids if i != 0}
    # Distâncias depot->i and i->j (use dist_km)
    depot_idx = id_to_index[0]
    # savings s_ij = d_0i + d_0j - d_ij
    savings = []
    for i in ids:
        if i == 0: continue
        for j in ids:
            if j == 0 or j == i: continue
            s = dist_km[depot_idx, id_to_index[i]] + dist_km[depot_idx, id_to_index[j]] - dist_km[id_to_index[i], id_to_index[j]]
            savings.append((s, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])
    # map client -> route key
    client_route_key = {i: i for i in ids if i != 0}
    for s, i, j in savings:
        ri_key = client_route_key.get(i)
        rj_key = client_route_key.get(j)
        if ri_key is None or rj_key is None or ri_key == rj_key:
            continue
        ri = routes[ri_key]
        rj = routes[rj_key]
        # verificar se i está na extremidade direita de ri (antes do 0) and j está na extremidade esquerda de rj (após o 0)
        # para poder concatenar ri + rj (sem duplicar depósito)
        if ri[-2] == i and rj[1] == j:
            new_load = route_loads[ri_key] + route_loads[rj_key]
            if new_load <= CAPACIDADE:
                # tentativa de rota concatenada
                new_route = ri[:-1] + rj[1:]
                # verificar tempo da nova rota
                dist_km_val, time_min_val = route_distance_and_time(new_route)
                if time_min_val <= TEMPO_MAX_DIA_MIN:
                    # fazer merge
                    new_key = ri_key  # manter key do primeiro
                    routes[new_key] = new_route
                    route_loads[new_key] = new_load
                    # remover rj
                    del routes[rj_key]
                    del route_loads[rj_key]
                    # atualizar client_route_key para todos clientes em rj
                    for client in rj:
                        if client != 0:
                            client_route_key[client] = new_key
    # Resultado em lista
    final_routes = list(routes.values())
    return final_routes

# -----------------------------
# Gerar rotas com cada heurística
# -----------------------------
print("Gerando rotas com heurísticas...")

routes_sweep = sweep_routes()
routes_nearest = nearest_neighbor_routes(farthest=False)
routes_farthest = nearest_neighbor_routes(farthest=True)
routes_cw = clarke_wright_routes()

all_solutions = {
    "Sweep": routes_sweep,
    "Nearest": routes_nearest,
    "Farthest": routes_farthest,
    "ClarkeWright": routes_cw,
}

# -----------------------------
# Função para imprimir resumo e agendar rotas por dia
# -----------------------------
def summarize_and_schedule(routes, name):
    print(f"\n=== Solução: {name} ===")
    summary = []
    for idx, r in enumerate(routes):
        dist_km_val, time_min_val = route_distance_and_time(r)
        load = route_load(r)
        summary.append({
            "index": idx,
            "route": r,
            "dist_km": dist_km_val,
            "time_min": time_min_val,
            "load": load,
        })
    # imprimir rotas
    for s in summary:
        readable = " -> ".join(clientes[n]["nome"] for n in s["route"])
        print(f"Rota {s['index']}: {readable}")
        print(f"   carga={s['load']} kg  dist={s['dist_km']:.2f} km  tempo={s['time_min']:.1f} min")
    # agendamento semanal (dias de 1..7) - greedy first-fit: preencher dias com rotas sem exceder TEMPO_MAX_DIA_MIN
    days = [[] for _ in range(7)]
    days_time = [0.0 for _ in range(7)]
    # ordenar rotas por tempo decrescente (best-fit decreasing para distribuir)
    sorted_summary = sorted(summary, key=lambda x: x["time_min"], reverse=True)
    for s in sorted_summary:
        placed = False
        # primeira tentativa: colocar na primeira data que caiba
        for d in range(7):
            if days_time[d] + s["time_min"] <= TEMPO_MAX_DIA_MIN:
                days[d].append(s)
                days_time[d] += s["time_min"]
                placed = True
                break
        if not placed:
            # se não coube em nenhuma, criar novo dia (mas temos só 7 dias na semana) -> marcar como overflow
            # para efeito do trabalho, vamos ainda adicionar na semana com menor tempo (mesmo que exceda) e notificar
            min_idx = int(np.argmin(days_time))
            days[min_idx].append(s)
            days_time[min_idx] += s["time_min"]
            print("Aviso: uma rota não coube em nenhum dia sem exceder 6h; forçando atribuição a dia de menor carga (excedente gerado).")
    # imprimir programação
    print("\nProgramação semanal (dias com rotas):")
    for d_idx, d in enumerate(days, start=1):
        if not d:
            continue
        print(f" Dia {d_idx}: tempo_total={days_time[d_idx-1]:.1f} min")
        for s in d:
            route_names = " -> ".join(clientes[n]["nome"] for n in s["route"])
            print(f"   Rota {s['index']}: carga={s['load']} kg tempo={s['time_min']:.1f} min dist={s['dist_km']:.2f} km")
    # Estatísticas gerais
    total_dist = sum(s["dist_km"] for s in summary)
    total_time = sum(s["time_min"] for s in summary)
    total_load = sum(s["load"] for s in summary)
    print(f"\nResumo {name}: rotas={len(summary)}  distancia_total={total_dist:.2f} km  tempo_total={total_time:.1f} min  carga_total_semana={total_load:.1f} kg")
    return {
        "summary": summary,
        "days": days,
        "days_time": days_time,
        "total_dist": total_dist,
        "total_time": total_time,
        "total_load": total_load,
    }

results = {}
for name, routes in all_solutions.items():
    results[name] = summarize_and_schedule(routes, name)

# -----------------------------
# Visualização básica (opcional)
# Plota o grafo e sobrepõe as rotas (usar a solução Clarke & Wright como exemplo)
# -----------------------------
try:
    if G is not None:
        sol_name = "ClarkeWright"
        sol = results[sol_name]["summary"]
        print(f"\nPlotando grafo com rotas da solução {sol_name}...")
        fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0, edge_color="gray", figsize=(10,10))
        # plotar pontos
        for i in ids:
            lon = clientes[i]["lon"]; lat = clientes[i]["lat"]
            if i == 0:
                ax.scatter(lon, lat, s=80, marker='*', zorder=5)
                ax.text(lon, lat, " DEPÓSITO", fontsize=8)
            else:
                ax.scatter(lon, lat, s=40, zorder=5)
                ax.text(lon, lat, f" {i}", fontsize=7)
        # plotar cada rota com linha entre coordenadas (usando caminho mais curto no grafo)
        colors = ["red","blue","green","orange","purple","brown","magenta","cyan"]
        for r_idx, r in enumerate(sol):
            route_nodes = r["route"]
            color = colors[r_idx % len(colors)]
            # construir lista de graph nodes correspondentes
            if nearest_node is not None:
                graph_path = []
                for a,b in zip(route_nodes[:-1], route_nodes[1:]):
                    u = nearest_node.get(a); v = nearest_node.get(b)
                    if u is not None and v is not None:
                        try:
                            sp = nx.shortest_path(G, u, v, weight="length")
                            graph_path.extend(sp)
                        except Exception:
                            # ignore path errors
                            pass
                # extrair coords dos nós do grafo
                # remover duplicatas consecutivas
                unique_path = []
                for node in graph_path:
                    if not unique_path or unique_path[-1] != node:
                        unique_path.append(node)
                if unique_path:
                    xs = [G.nodes[n]['x'] for n in unique_path]
                    ys = [G.nodes[n]['y'] for n in unique_path]
                    ax.plot(xs, ys, linewidth=2, alpha=0.8, color=color, zorder=3)
            else:
                # plotar linhas diretas entre coordenadas dos clientes
                xs = [clientes[node]["lon"] for node in route_nodes]
                ys = [clientes[node]["lat"] for node in route_nodes]
                ax.plot(xs, ys, linewidth=2, alpha=0.8, color=color, zorder=3)
        plt.title(f"Rotas ({sol_name}) sobre grafo de Brasília")
        plt.show()
    else:
        print("\nVisualizacao do grafo OSM nao disponivel (usando apenas coordenadas)")
        # Plot simples com matplotlib sem o grafo OSM
        sol_name = "ClarkeWright"
        sol = results[sol_name]["summary"]
        fig, ax = plt.subplots(figsize=(10, 8))
        # plotar pontos
        for i in ids:
            lon = clientes[i]["lon"]; lat = clientes[i]["lat"]
            if i == 0:
                ax.scatter(lon, lat, s=100, marker='*', color='red', zorder=5)
                ax.text(lon, lat, " DEPÓSITO", fontsize=10)
            else:
                ax.scatter(lon, lat, s=60, color='blue', zorder=5)
                ax.text(lon, lat, f" {i}", fontsize=8)
        # plotar rotas com linhas diretas
        colors = ["red","blue","green","orange","purple","brown","magenta","cyan"]
        for r_idx, r in enumerate(sol):
            route_nodes = r["route"]
            color = colors[r_idx % len(colors)]
            xs = [clientes[node]["lon"] for node in route_nodes]
            ys = [clientes[node]["lat"] for node in route_nodes]
            ax.plot(xs, ys, linewidth=2, alpha=0.8, color=color, zorder=3)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.title(f"Rotas ({sol_name}) - Brasília")
        plt.grid(True, alpha=0.3)
        plt.show()
except Exception as e:
    print("Plot falhou:", e)

print("\nExecução finalizada. Consulte as rotas e programação impressas acima.")
