"""
run_optimized.py - Versão otimizada com visualização aprimorada

Gera rotas usando o grafo real de Brasília com:
- Bounding box ajustado aos pontos de entrega
- Caminhos traçados nos nós reais do graphml
- Visualização em imagem e HTML interativo
"""

import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import folium
from folium import plugins

# -----------------------------
# Dados e parâmetros
# -----------------------------
CAPACIDADE = 1800  # kg
VEL_KMH = 50.0
TEMPO_MAX_DIA_MIN = 6 * 60

# Dados de demanda e tempo de descarga (não estão no CSV)
dados_demanda = {
    "Depósito (Carrefour STN)": {"demanda": 0, "descarga": 515},
    "CLS 307": {"demanda": 160, "descarga": 50},
    "CLS 114": {"demanda": 170, "descarga": 60},
    "CLN 110": {"demanda": 22, "descarga": 70},
    "SOF (Água Mineral)": {"demanda": 300, "descarga": 85},
    "SHIS QI 17 (Lago Sul)": {"demanda": 250, "descarga": 45},
    "CLSW 103": {"demanda": 90, "descarga": 65},
    "Varjão (entrada)": {"demanda": 130, "descarga": 55},
    "Águas Claras (shopping)": {"demanda": 350, "descarga": 40},
    "Taguatinga Pistão Sul": {"demanda": 900, "descarga": 45},
}

# Carregar coordenadas do CSV
print("Carregando pontos do CSV...")
pontos_df = pd.read_csv("data/pontos.csv")
print(f"✓ {len(pontos_df)} pontos carregados do CSV")

# Criar dicionário de clientes combinando CSV + dados de demanda
clientes = {}
for idx, row in pontos_df.iterrows():
    nome = str(row["Nome"])
    clientes[idx] = {
        "nome": nome,
        "lat": row["Latitude"],
        "lon": row["Longitude"],
        "demanda": dados_demanda[nome]["demanda"],
        "descarga": dados_demanda[nome]["descarga"],
    }

ids = sorted(clientes.keys())
print(f"Pontos de entrega:")
for i in ids:
    print(
        f"  {i}: {clientes[i]['nome']} ({clientes[i]['lat']:.6f}, {clientes[i]['lon']:.6f})"
    )


# -----------------------------
# Funções utilitárias
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def travel_time_minutes_km(distance_km, vel_kmh=VEL_KMH):
    return (distance_km / vel_kmh) * 60.0


def route_service_time_minutes(route):
    return sum(clientes[node_id]["descarga"] for node_id in route if node_id != 0)


# -----------------------------
# Carregar grafo e mapear pontos
# -----------------------------
print("Carregando grafo de Brasília...")
G = ox.load_graphml("data/brasilia.graphml")
print(f"✓ Grafo carregado: {len(G.nodes)} nós, {len(G.edges)} arestas")

# Calcular bounding box baseado nos pontos de entrega
lats = [clientes[i]["lat"] for i in ids]
lons = [clientes[i]["lon"] for i in ids]
margin = 0.01  # margem pequena
bbox = {
    "north": max(lats) + margin,
    "south": min(lats) - margin,
    "east": max(lons) + margin,
    "west": min(lons) - margin,
}
print(
    f"Bounding box: N={bbox['north']:.4f}, S={bbox['south']:.4f}, E={bbox['east']:.4f}, W={bbox['west']:.4f}"
)

# Mapear clientes para nós mais próximos do grafo
print("Mapeando clientes para nós do grafo...")
nearest_node = {}
for i in ids:
    lat, lon = clientes[i]["lat"], clientes[i]["lon"]
    # Encontrar nó mais próximo manualmente (fallback sem scikit-learn)
    min_dist = float("inf")
    closest = None
    for node, data in G.nodes(data=True):
        dist = haversine_km(lat, lon, data["y"], data["x"])
        if dist < min_dist:
            min_dist = dist
            closest = node
    nearest_node[i] = closest
    print(
        f"  Cliente {i} ({clientes[i]['nome'][:20]}...) -> nó {closest} (dist={min_dist*1000:.1f}m)"
    )

# -----------------------------
# Calcular matriz de distâncias usando shortest path
# -----------------------------
print("\nCalculando matriz de distâncias no grafo real...")
n = len(ids)
dist_km = np.zeros((n, n))
time_min = np.zeros((n, n))
shortest_paths = {}  # Armazenar caminhos para visualização

id_to_index = {node_id: idx for idx, node_id in enumerate(ids)}

for i_idx, i in enumerate(ids):
    for j_idx, j in enumerate(ids):
        if i == j:
            continue

        u = nearest_node[i]
        v = nearest_node[j]
        try:
            length_m = nx.shortest_path_length(G, u, v, weight="length")
            path = nx.shortest_path(G, u, v, weight="length")
            shortest_paths[(i, j)] = path
            dist_km[i_idx, j_idx] = length_m / 1000.0
            time_min[i_idx, j_idx] = travel_time_minutes_km(length_m / 1000.0)
        except nx.NetworkXNoPath:
            # Fallback para haversine
            km = haversine_km(
                clientes[i]["lat"],
                clientes[i]["lon"],
                clientes[j]["lat"],
                clientes[j]["lon"],
            )
            dist_km[i_idx, j_idx] = km
            time_min[i_idx, j_idx] = travel_time_minutes_km(km)

print("✓ Matriz de distâncias calculada")


# -----------------------------
# Funções de avaliação de rotas
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


# -----------------------------
# Otimização: Remoção de Cruzamentos (2-opt)
# -----------------------------


def remove_crossings(route):
    """
    Remove cruzamentos de uma rota usando o algoritmo 2-opt.
    O 2-opt detecta quando duas arestas se cruzam e as troca para eliminar o cruzamento.
    
    Args:
        route: Lista de IDs de clientes na rota (incluindo depósito no início e fim)
    
    Returns:
        Rota otimizada sem cruzamentos
    """
    if len(route) <= 3:  # Rota muito pequena, não precisa otimizar
        return route
    
    improved = True
    best_route = route[:]
    
    while improved:
        improved = False
        best_distance, _ = route_distance_and_time(best_route)
        
        # Tentar trocar todos os pares de arestas
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                # Criar nova rota invertendo o segmento entre i e j
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                
                # Verificar se a nova rota é melhor
                new_distance, _ = route_distance_and_time(new_route)
                
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
                    break
            
            if improved:
                break
    
    return best_route


def detect_crossing(p1, p2, p3, p4):
    """
    Detecta se duas linhas (p1-p2 e p3-p4) se cruzam.
    Usa o método de orientação geométrica.
    
    Args:
        p1, p2: Pontos (lat, lon) da primeira linha
        p3, p4: Pontos (lat, lon) da segunda linha
    
    Returns:
        True se as linhas se cruzam, False caso contrário
    """
    def ccw(A, B, C):
        """Verifica se três pontos estão em sentido anti-horário"""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def count_crossings_in_route(route):
    """
    Conta o número de cruzamentos em uma rota.
    
    Args:
        route: Lista de IDs de clientes na rota
    
    Returns:
        Número de cruzamentos detectados
    """
    crossings = 0
    
    # Obter coordenadas de todos os pontos da rota
    coords = [(clientes[node]["lat"], clientes[node]["lon"]) for node in route]
    
    # Verificar cada par de arestas
    for i in range(len(route) - 1):
        for j in range(i + 2, len(route) - 1):
            # Evitar comparar arestas adjacentes
            if j == i + 1:
                continue
            
            p1 = coords[i]
            p2 = coords[i + 1]
            p3 = coords[j]
            p4 = coords[j + 1]
            
            if detect_crossing(p1, p2, p3, p4):
                crossings += 1
    
    return crossings


# -----------------------------
# Heurísticas de Roteirização
# -----------------------------


# 1) Nearest Neighbor (Vizinho Mais Próximo)
def nearest_neighbor_routes():
    """Constrói rotas sempre escolhendo o cliente mais próximo não visitado."""
    routes = []
    remaining = set(ids) - {0}

    while remaining:
        route = [0]
        current_load = 0
        current_time = 0
        current = 0

        while remaining:
            best = None
            best_dist = float("inf")

            for candidate in remaining:
                i_curr = id_to_index[current]
                i_cand = id_to_index[candidate]

                dist = dist_km[i_curr, i_cand]
                if dist < best_dist:
                    best_dist = dist
                    best = candidate

            if best is None:
                break

            route.append(best)
            current_load += clientes[best]["demanda"]
            i_curr = id_to_index[current]
            i_best = id_to_index[best]
            current_time += time_min[i_curr, i_best] + clientes[best]["descarga"]
            current = best
            remaining.remove(best)

        route.append(0)
        
        # Aplicar otimização 2-opt para remover cruzamentos
        route = remove_crossings(route)
        
        routes.append(route)

    return routes


# 2) Farthest Neighbor (Ponto Mais Distante)
def farthest_neighbor_routes():
    """Sempre escolhe o cliente mais distante não visitado."""
    routes = []
    remaining = set(ids) - {0}

    while remaining:
        route = [0]
        current_load = 0
        current_time = 0
        current = 0

        while remaining:
            # Encontrar o cliente mais distante do ponto atual
            best = None
            best_dist = -1  # Queremos o MAIOR

            for candidate in remaining:
                i_curr = id_to_index[current]
                i_cand = id_to_index[candidate]
                dist = dist_km[i_curr, i_cand]
                
                if dist > best_dist:
                    best_dist = dist
                    best = candidate

            if best is None:
                break

            route.append(best)
            current_load += clientes[best]["demanda"]
            i_curr = id_to_index[current]
            i_best = id_to_index[best]
            current_time += time_min[i_curr, i_best] + clientes[best]["descarga"]
            current = best
            remaining.remove(best)

        route.append(0)
        routes.append(route)

    return routes


# 3) Sweep (Varredura por ângulo)
def sweep_routes():
    """Ordena clientes por ângulo e agrupa sequencialmente."""
    depot = clientes[0]
    angles = []
    for i in ids:
        if i == 0:
            continue
        dx = clientes[i]["lon"] - depot["lon"]
        dy = clientes[i]["lat"] - depot["lat"]
        ang = math.atan2(dy, dx)
        angles.append((i, ang))
    angles.sort(key=lambda x: x[1])

    routes = []
    current_route = [0]
    current_load = 0
    current_time = 0

    for i, ang in angles:
        demand = clientes[i]["demanda"]

        if current_route == [0]:
            current_route.append(i)
            current_load += demand
            i_depot = id_to_index[0]
            i_curr = id_to_index[i]
            current_time = (
                time_min[i_depot, i_curr]
                + clientes[i]["descarga"]
                + time_min[i_curr, i_depot]
            )
        else:
            last = current_route[-1]
            i_last = id_to_index[last]
            i_curr = id_to_index[i]
            i_depot = id_to_index[0]

            time_to_new = time_min[i_last, i_curr]
            time_new_to_depot = time_min[i_curr, i_depot]
            time_old_to_depot = time_min[i_last, i_depot]

            additional_time = (
                time_to_new
                + clientes[i]["descarga"]
                + time_new_to_depot
                - time_old_to_depot
            )
            new_time = current_time + additional_time

            if current_load + demand <= CAPACIDADE and new_time <= TEMPO_MAX_DIA_MIN:
                current_route.append(i)
                current_load += demand
                current_time = new_time
            else:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0, i]
                current_load = demand
                i_depot = id_to_index[0]
                i_curr = id_to_index[i]
                current_time = (
                    time_min[i_depot, i_curr]
                    + clientes[i]["descarga"]
                    + time_min[i_curr, i_depot]
                )

    if len(current_route) > 1:
        current_route.append(0)
        routes.append(current_route)

    return routes


# 4) Heurística Clarke & Wright (Savings)
def clarke_wright_routes():
    routes = {i: [0, i, 0] for i in ids if i != 0}
    route_loads = {i: clientes[i]["demanda"] for i in ids if i != 0}
    depot_idx = id_to_index[0]
    savings = []

    for i in ids:
        if i == 0:
            continue
        for j in ids:
            if j == 0 or j == i:
                continue
            s = (
                dist_km[depot_idx, id_to_index[i]]
                + dist_km[depot_idx, id_to_index[j]]
                - dist_km[id_to_index[i], id_to_index[j]]
            )
            savings.append((s, i, j))

    savings.sort(reverse=True, key=lambda x: x[0])
    client_route_key = {i: i for i in ids if i != 0}

    for s, i, j in savings:
        ri_key = client_route_key.get(i)
        rj_key = client_route_key.get(j)
        if ri_key is None or rj_key is None or ri_key == rj_key:
            continue

        ri = routes[ri_key]
        rj = routes[rj_key]

        if ri[-2] == i and rj[1] == j:
            new_load = route_loads[ri_key] + route_loads[rj_key]
            if new_load <= CAPACIDADE:
                new_route = ri[:-1] + rj[1:]
                dist_km_val, time_min_val = route_distance_and_time(new_route)
                if time_min_val <= TEMPO_MAX_DIA_MIN:
                    new_key = ri_key
                    routes[new_key] = new_route
                    route_loads[new_key] = new_load
                    del routes[rj_key]
                    del route_loads[rj_key]
                    for client in rj:
                        if client != 0:
                            client_route_key[client] = new_key

    return list(routes.values())


# -----------------------------
# Gerar rotas com todos os 4 algoritmos
# -----------------------------
print("\n" + "=" * 60)
print("EXECUTANDO 4 HEURÍSTICAS DE ROTEIRIZAÇÃO")
print("=" * 60)

algorithms = [
    ("Clarke & Wright", clarke_wright_routes),
    ("Vizinho Mais Próximo", nearest_neighbor_routes),
    ("Ponto Mais Distante", farthest_neighbor_routes),
    ("Varredura (Sweep)", sweep_routes),
]

all_results = {}

for algo_name, algo_func in algorithms:
    print(f"\n\nGerando rotas com {algo_name}...")
    routes = algo_func()

    route_summary = []
    total_dist = 0
    total_load = 0
    total_crossings = 0

    for idx, r in enumerate(routes):
        dist, time = route_distance_and_time(r)
        load = route_load(r)
        crossings = count_crossings_in_route(r)
        total_crossings += crossings
        
        route_summary.append(
            {"index": idx, "route": r, "dist_km": dist, "time_min": time, "load": load, "crossings": crossings}
        )
        readable = " → ".join(clientes[n]["nome"] for n in r)
        print(f"\nRota {idx+1}: {readable}")
        print(
            f"  Carga: {load} kg ({load/CAPACIDADE*100:.1f}%) | Distância: {dist:.2f} km | Tempo: {time:.1f} min ({time/TEMPO_MAX_DIA_MIN*100:.1f}%)"
        )
        if crossings > 0:
            print(f"  ⚠️  Cruzamentos detectados: {crossings}")
        total_dist += dist
        total_load += load

    print(f"\n--- RESUMO {algo_name.upper()} ---")
    print(f"Número de rotas: {len(routes)}")
    print(f"Distância total: {total_dist:.2f} km")
    print(f"Carga total: {total_load} kg")
    print(f"Total de cruzamentos: {total_crossings}")

    all_results[algo_name] = {
        "routes": routes,
        "route_summary": route_summary,
        "total_dist": total_dist,
        "total_load": total_load,
        "num_routes": len(routes),
        "total_crossings": total_crossings,
    }

# -----------------------------
# Exportar Matrizes para CSV
# -----------------------------
print("\n📄 Exportando matrizes para CSV...")

# 1. Matriz de Distâncias
print("  - Matriz de distâncias (dist_matrix.csv)")
dist_matrix_df = pd.DataFrame(
    dist_km,
    columns=[clientes[i]["nome"] for i in ids],
    index=[clientes[i]["nome"] for i in ids]
)
dist_matrix_df.to_csv("output/dist_matrix.csv")

# 2. Matriz de Ganhos (Clarke & Wright Savings)
print("  - Matriz de ganhos (savings_matrix.csv)")
depot_idx = id_to_index[0]
savings_matrix = np.zeros((n, n))
savings_list = []

for i_idx, i in enumerate(ids):
    if i == 0:
        continue
    for j_idx, j in enumerate(ids):
        if j == 0 or j == i:
            continue
        # Saving = dist(depot, i) + dist(depot, j) - dist(i, j)
        saving = (
            dist_km[depot_idx, i_idx]
            + dist_km[depot_idx, j_idx]
            - dist_km[i_idx, j_idx]
        )
        savings_matrix[i_idx, j_idx] = saving
        savings_list.append({
            "Cliente_i": clientes[i]["nome"],
            "Cliente_j": clientes[j]["nome"],
            "Dist_Deposito_i": dist_km[depot_idx, i_idx],
            "Dist_Deposito_j": dist_km[depot_idx, j_idx],
            "Dist_i_j": dist_km[i_idx, j_idx],
            "Ganho": saving
        })

savings_matrix_df = pd.DataFrame(
    savings_matrix,
    columns=[clientes[i]["nome"] for i in ids],
    index=[clientes[i]["nome"] for i in ids]
)
savings_matrix_df.to_csv("output/savings_matrix.csv")

# 3. Hierarquia de Ganhos (ordenada por ganho decrescente)
print("  - Hierarquia de ganhos (savings_hierarchy.csv)")
savings_hierarchy_df = pd.DataFrame(savings_list)
savings_hierarchy_df = savings_hierarchy_df.sort_values(by="Ganho", ascending=False)
savings_hierarchy_df["Rank"] = range(1, len(savings_hierarchy_df) + 1)
savings_hierarchy_df = savings_hierarchy_df[["Rank", "Cliente_i", "Cliente_j", "Dist_Deposito_i", "Dist_Deposito_j", "Dist_i_j", "Ganho"]]
savings_hierarchy_df.to_csv("output/savings_hierarchy.csv", index=False)

print("✓ Matrizes exportadas com sucesso!")

# Usar Clarke & Wright para visualização individual (compatibilidade com código existente)
routes = all_results["Clarke & Wright"]["routes"]
route_summary = all_results["Clarke & Wright"]["route_summary"]

# -----------------------------
# Visualização 1: Imagem PNG com bounding box ajustado
# -----------------------------
print("\n📊 Gerando visualização em imagem...")

# Extrair subgrafo dentro do bounding box
nodes_in_bbox = []
for node, data in G.nodes(data=True):
    if (
        bbox["south"] <= data["y"] <= bbox["north"]
        and bbox["west"] <= data["x"] <= bbox["east"]
    ):
        nodes_in_bbox.append(node)

G_bbox = G.subgraph(nodes_in_bbox)
print(f"Subgrafo: {len(G_bbox.nodes)} nós, {len(G_bbox.edges)} arestas")

# Convert to MultiDiGraph if needed for plotting
if not isinstance(G_bbox, (nx.MultiGraph, nx.MultiDiGraph)):
    G_bbox = nx.MultiDiGraph(G_bbox)

fig, ax = ox.plot_graph(
    G_bbox,
    node_size=0,
    edge_color="#CCCCCC",
    edge_linewidth=0.5,
    bgcolor="white",
    show=False,
    close=False,
    figsize=(16, 12),
)

# Cores para as rotas
colors = ["#FF0000", "#0000FF", "#00AA00", "#FF8800", "#8800FF", "#00FFFF"]

# Plotar rotas com caminhos reais
for r_idx, r_info in enumerate(route_summary):
    route = r_info["route"]
    color = colors[r_idx % len(colors)]

    # Traçar caminho real usando os nós do grafo
    for a, b in zip(route[:-1], route[1:]):
        if (a, b) in shortest_paths:
            path_nodes = shortest_paths[(a, b)]
            xs = [G.nodes[node]["x"] for node in path_nodes]
            ys = [G.nodes[node]["y"] for node in path_nodes]
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=3,
                alpha=0.7,
                zorder=2,
                label=f"Rota {r_idx}" if a == route[0] and b == route[1] else "",
            )

# Plotar pontos de clientes
for i in ids:
    lon, lat = clientes[i]["lon"], clientes[i]["lat"]
    if i == 0:
        ax.scatter(
            lon,
            lat,
            s=400,
            marker="*",
            color="red",
            edgecolors="black",
            linewidths=2,
            zorder=5,
            label="Depósito",
        )
    else:
        ax.scatter(
            lon,
            lat,
            s=200,
            marker="o",
            color="yellow",
            edgecolors="black",
            linewidths=2,
            zorder=5,
        )
    # Label
    ax.annotate(
        clientes[i]["nome"],
        xy=(lon, lat),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

ax.set_title(
    "Rotas de Entrega - Brasília (Clarke & Wright)\nCaminhos Reais sobre Rede Viária OSM",
    fontsize=16,
    fontweight="bold",
)
ax.legend(loc="upper left", fontsize=10)

plt.tight_layout()
plt.savefig("output/rotas_brasilia.png", dpi=300, bbox_inches="tight")
print("✓ Imagem salva: rotas_brasilia.png")
plt.close()

# -----------------------------
# Visualização 2: Comparação dos 4 Algoritmos em uma única figura
# -----------------------------
print("\n📊 Gerando comparação dos 4 algoritmos...")

# Criar figura com 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

# Cores para as rotas
route_colors = ["#FF0000", "#0000FF", "#00AA00", "#FF8800", "#8800FF", "#00FFFF"]

for idx, (algo_name, result) in enumerate(all_results.items()):
    ax = axes[idx]

    # Plotar pontos de clientes
    for i in ids:
        lon, lat = clientes[i]["lon"], clientes[i]["lat"]
        if i == 0:
            ax.scatter(
                lon,
                lat,
                s=500,
                marker="*",
                color="red",
                edgecolors="black",
                linewidths=2,
                zorder=5,
            )
        else:
            ax.scatter(
                lon,
                lat,
                s=250,
                marker="o",
                color="yellow",
                edgecolors="black",
                linewidths=2,
                zorder=5,
            )

        # Labels mais compactos
        nome_curto = clientes[i]["nome"].split("(")[0].strip()[:15]
        ax.annotate(
            nome_curto,
            xy=(lon, lat),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    # Plotar rotas
    for r_idx, r_info in enumerate(result["route_summary"]):
        route = r_info["route"]
        color = route_colors[r_idx % len(route_colors)]

        # Desenhar linhas conectando os clientes
        for a, b in zip(route[:-1], route[1:]):
            lon_a, lat_a = clientes[a]["lon"], clientes[a]["lat"]
            lon_b, lat_b = clientes[b]["lon"], clientes[b]["lat"]
            ax.plot(
                [lon_a, lon_b],
                [lat_a, lat_b],
                color=color,
                linewidth=2.5,
                alpha=0.7,
                zorder=2,
            )
            # Adicionar setas para indicar direção
            ax.annotate(
                "",
                xy=(lon_b, lat_b),
                xytext=(lon_a, lat_a),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.6),
            )

    # Título do subplot com estatísticas
    title = f"{algo_name}\n"
    title += f"Rotas: {result['num_routes']} | Dist: {result['total_dist']:.1f} km"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Configurações do eixo
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal", "box")

    # Adicionar legenda das rotas no subplot
    legend_elements = []
    for r_idx, r_info in enumerate(result["route_summary"]):
        color = route_colors[r_idx % len(route_colors)]
        from matplotlib.lines import Line2D

        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2,
                label=f'R{r_idx+1}: {r_info["dist_km"]:.1f}km, {r_info["load"]}kg',
            )
        )
    ax.legend(handles=legend_elements, loc="lower left", fontsize=7, framealpha=0.9)

# Título geral da figura
fig.suptitle(
    "Comparação de Heurísticas de Roteirização - Brasília",
    fontsize=18,
    fontweight="bold",
    y=0.995,
)

plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))
plt.savefig("output/comparacao_algoritmos.png", dpi=300, bbox_inches="tight")
print("✓ Imagem de comparação salva: comparacao_algoritmos.png")
plt.close()

# -----------------------------
# Visualização 2b: Gráficos de Barras Comparativos
# -----------------------------
print("\n📊 Gerando gráficos de barras comparativos...")

# Preparar dados
algoritmos_names = list(all_results.keys())
algoritmos_labels = [name.replace(" ", "\n") for name in algoritmos_names]
distancias_totais = [all_results[name]["total_dist"] for name in algoritmos_names]
cruzamentos_totais = [all_results[name]["total_crossings"] for name in algoritmos_names]
cores_bar = ['#2E7D32', '#C62828', '#F57C00', '#1565C0']

# Gráfico 1: Distância Total e Cruzamentos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Distância Total
x_pos = np.arange(len(algoritmos_labels))
bars1 = ax1.bar(x_pos, distancias_totais, color=cores_bar, alpha=0.8, edgecolor='black', linewidth=1.5)

# Adicionar valores nas barras
for i, (bar, dist) in enumerate(zip(bars1, distancias_totais)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{dist:.2f} km',
             ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Adicionar percentual de diferença em relação ao melhor
    min_dist = min(distancias_totais)
    if dist != min_dist:
        diff_pct = ((dist - min_dist) / min_dist) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                 f'+{diff_pct:.1f}%',
                 ha='center', va='center', fontsize=9, color='white', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

ax1.set_ylabel('Distância Total (km)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Algoritmo', fontsize=12, fontweight='bold')
ax1.set_title('Comparação de Distância Total por Algoritmo', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(algoritmos_labels, fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(distancias_totais) * 1.15)

# Destacar o melhor resultado
min_idx = distancias_totais.index(min(distancias_totais))
bars1[min_idx].set_edgecolor('gold')
bars1[min_idx].set_linewidth(3)
ax1.text(min_idx, distancias_totais[min_idx] * 1.05, '⭐ MELHOR', 
         ha='center', fontsize=10, fontweight='bold', color='#FFD700')

# Subplot 2: Cruzamentos
bars2 = ax2.bar(x_pos, cruzamentos_totais, color=cores_bar, alpha=0.8, edgecolor='black', linewidth=1.5)

# Adicionar valores nas barras
for i, (bar, cruz) in enumerate(zip(bars2, cruzamentos_totais)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{cruz}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

ax2.set_ylabel('Número de Cruzamentos', fontsize=12, fontweight='bold')
ax2.set_xlabel('Algoritmo', fontsize=12, fontweight='bold')
ax2.set_title('Cruzamentos Detectados por Algoritmo', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(algoritmos_labels, fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, max(cruzamentos_totais) * 1.2 if max(cruzamentos_totais) > 0 else 2)

# Destacar o melhor resultado (menos cruzamentos)
min_cross_idx = cruzamentos_totais.index(min(cruzamentos_totais))
bars2[min_cross_idx].set_edgecolor('gold')
bars2[min_cross_idx].set_linewidth(3)
if cruzamentos_totais[min_cross_idx] == 0:
    ax2.text(min_cross_idx, 0.3, '✅ SEM\nCRUZAMENTOS', 
             ha='center', fontsize=9, fontweight='bold', color='green')

plt.suptitle('Análise Comparativa de Heurísticas de Roteirização - Brasília', 
             fontsize=16, fontweight='bold', y=1.00)

plt.tight_layout()
plt.savefig('output/grafico_comparacao_barras.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico de barras salvo: grafico_comparacao_barras.png")
plt.close()

# Gráfico 2: Análise Detalhada das Rotas
print("  Gerando análise detalhada das rotas...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (algo_name, result) in enumerate(all_results.items()):
    ax = axes[idx]
    
    # Dados das rotas
    route_summary_detail = result["route_summary"]
    rotas_nomes = [f'Rota {i+1}' for i in range(len(route_summary_detail))]
    distancias_rotas = [r["dist_km"] for r in route_summary_detail]
    cargas = [r["load"] for r in route_summary_detail]
    tempos = [r["time_min"] for r in route_summary_detail]
    
    # Criar gráfico de barras agrupadas
    x = np.arange(len(rotas_nomes))
    width = 0.25
    
    bars1 = ax.bar(x - width, distancias_rotas, width, label='Distância (km)', 
                   color=cores_bar[idx], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, [c/10 for c in cargas], width, label='Carga (x100 kg)', 
                   color='orange', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, [t/10 for t in tempos], width, label='Tempo (x10 min)', 
                   color='purple', alpha=0.8, edgecolor='black')
    
    # Adicionar valores nas barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Valores', fontsize=10, fontweight='bold')
    ax.set_title(f'{algo_name}\nDistância Total: {sum(distancias_rotas):.2f} km | Cruzamentos: {result["total_crossings"]}', 
                fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(rotas_nomes)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle('Análise Detalhada das Rotas por Algoritmo', 
             fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('output/analise_detalhada_rotas.png', dpi=300, bbox_inches='tight')
print("✓ Análise detalhada salva: analise_detalhada_rotas.png")
plt.close()

# -----------------------------
# Visualização 3: Mapas HTML interativos (um para cada algoritmo)
# -----------------------------
print("\n🗺️  Gerando mapas HTML interativos para cada algoritmo...")

# Centro do mapa
center_lat = sum(lats) / len(lats)
center_lon = sum(lons) / len(lons)

for algo_name, result in all_results.items():
    # Nome do arquivo (substituir espaços e caracteres especiais)
    filename = algo_name.lower()
    filename = filename.replace(" ", "_")
    filename = filename.replace("&", "e")
    filename = filename.replace("â", "a")
    filename = filename.replace("ã", "a")
    filename = filename.replace("ó", "o")
    filename = filename.replace("(", "")
    filename = filename.replace(")", "")
    filename = f"{filename}_brasilia.html"
    
    # Criar mapa
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")
    
    routes_for_map = result["routes"]
    route_summary_for_map = result["route_summary"]
    
    # Adicionar rotas com caminhos reais
    for r_idx, r_info in enumerate(route_summary_for_map):
        route = r_info["route"]
        color = colors[r_idx % len(colors)]

        # Construir caminho completo
        full_path_coords = []
        for a, b in zip(route[:-1], route[1:]):
            if (a, b) in shortest_paths:
                path_nodes = shortest_paths[(a, b)]
                for node in path_nodes:
                    full_path_coords.append([G.nodes[node]["y"], G.nodes[node]["x"]])

        # Adicionar linha da rota
        if full_path_coords:
            folium.PolyLine(
                full_path_coords,
                color=color,
                weight=4,
                opacity=0.8,
                popup=f"Rota {r_idx+1}: {r_info['dist_km']:.1f}km, {r_info['load']}kg",
            ).add_to(m)

    # Adicionar marcadores dos clientes
    for i in ids:
        lat, lon = clientes[i]["lat"], clientes[i]["lon"]

        if i == 0:
            folium.Marker(
                [lat, lon],
                popup=f"<b>{clientes[i]['nome']}</b><br>DEPÓSITO",
                icon=folium.Icon(color="red", icon="home", prefix="fa"),
                tooltip=clientes[i]["nome"],
            ).add_to(m)
        else:
            # Determinar em qual rota está
            route_num = None
            for r_idx, r_info in enumerate(route_summary_for_map):
                if i in r_info["route"]:
                    route_num = r_idx + 1
                    break

            folium.CircleMarker(
                [lat, lon],
                radius=8,
                popup=f"<b>{clientes[i]['nome']}</b><br>Demanda: {clientes[i]['demanda']} kg<br>Rota: {route_num}",
                color="black",
                fillColor="yellow",
                fillOpacity=0.9,
                weight=2,
                tooltip=clientes[i]["nome"],
            ).add_to(m)

    # Adicionar legenda
    legend_html = f"""
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 300px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4 style="margin-top:0">Rotas de Entrega - Brasília</h4>
    <p><b>Algoritmo:</b> {algo_name}</p>
    <p><b>Total de rotas:</b> {len(routes_for_map)}</p>
    <p><b>Distância total:</b> {result['total_dist']:.1f} km</p>
    <p><b>Carga total:</b> {result['total_load']} kg</p>
    <p><b>Cruzamentos:</b> {result['total_crossings']}</p>
    <hr>
    """

    for r_idx, r_info in enumerate(route_summary_for_map):
        color = colors[r_idx % len(colors)]
        legend_html += f"""
        <p style="margin:5px 0">
            <span style="background-color:{color}; padding:2px 8px; color:white; font-weight:bold">
                Rota {r_idx+1}
            </span><br>
            {r_info["dist_km"]:.1f} km | {r_info["load"]} kg | {r_info["time_min"]:.0f} min
        </p>
        """

    legend_html += "</div>"
    from branca.element import Element

    m.get_root().add_child(Element(legend_html))

    # Salvar mapa
    m.save(f"output/{filename}")
    print(f"  ✓ {filename}")

print("\n" + "=" * 60)
print("✅ CONCLUÍDO!")
print("=" * 60)
print(f"📁 Arquivos gerados:")
print(f"\n  Imagens:")
print(f"   • comparacao_algoritmos.png - Comparação visual dos 4 algoritmos")
print(f"   • grafico_comparacao_barras.png - Gráficos de barras comparativos")
print(f"   • analise_detalhada_rotas.png - Análise detalhada por rota")
print(f"   • rotas_brasilia.png  - Imagem de alta resolução (Clarke & Wright)")
print(f"\n  Mapas HTML Interativos:")
for algo_name in all_results.keys():
    filename = algo_name.lower()
    filename = filename.replace(" ", "_").replace("&", "e").replace("â", "a").replace("ã", "a").replace("ó", "o").replace("(", "").replace(")", "")
    print(f"   • {filename}_brasilia.html - {algo_name}")
print(f"\n  Matrizes CSV:")
print(f"   • dist_matrix.csv - Matriz de distâncias (km)")
print(f"   • savings_matrix.csv - Matriz de ganhos Clarke & Wright")
print(f"   • savings_hierarchy.csv - Hierarquia de ganhos ordenada")
print(f"\n📊 Resumo da Comparação:")
for algo_name, result in all_results.items():
    print(f"   {algo_name}:")
    print(
        f"      - Rotas: {result['num_routes']} | Dist: {result['total_dist']:.2f} km | Carga: {result['total_load']} kg | Cruzamentos: {result['total_crossings']}"
    )
print("=" * 60)
