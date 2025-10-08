"""
run_with_csv_data.py - Versão que usa dados dos arquivos CSV

Script que carrega dados dos arquivos CSV em /data:
- pontos.csv: coordenadas dos pontos
- distancias.csv: matriz de distâncias reais
- matriz_pontos_com_links.csv: links para rotas OSM

Autor: Gabriel (versão com dados CSV)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# Dados e parâmetros
# -----------------------------
CAPACIDADE = 1800  # kg
VEL_KMH = 50.0  # velocidade média (km/h)
TEMPO_MAX_DIA_MIN = 6 * 60  # 6 horas em minutos

# Dados de demanda e tempo de descarga (não estão no CSV, usando valores do problema original)
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

# -----------------------------
# Carregar dados dos CSVs
# -----------------------------
def load_data():
    """Carrega dados dos arquivos CSV."""
    
    # Carregar pontos (coordenadas)
    pontos_df = pd.read_csv('data/pontos.csv')
    print("✓ Pontos carregados do CSV")
    
    # Carregar matriz de distâncias
    dist_df = pd.read_csv('data/distancias.csv', sep=';', index_col=0)
    print("✓ Matriz de distâncias carregada do CSV")
    
    # Criar dicionário de clientes combinando os dados
    clientes = {}
    for idx, row in pontos_df.iterrows():
        nome = row['Nome']
        clientes[idx] = {
            "nome": nome,
            "lat": row['Latitude'],
            "lon": row['Longitude'],
            "demanda": dados_demanda[nome]["demanda"],
            "descarga": dados_demanda[nome]["descarga"]
        }
    
    # Converter matriz de distâncias para numpy array
    dist_matrix = dist_df.values.astype(float)
    
    return clientes, dist_matrix, list(pontos_df['Nome'])

# Carregar dados
clientes, dist_km, nomes_pontos = load_data()
ids = list(range(len(clientes)))

print(f"\nPontos carregados: {len(clientes)}")
for i, nome in enumerate(nomes_pontos):
    print(f"  {i}: {nome}")

print(f"\nMatriz de distâncias (km):")
print(np.round(dist_km, 1))

# -----------------------------
# Criar matriz de tempo a partir das distâncias
# -----------------------------
def travel_time_minutes_km(distance_km, vel_kmh=VEL_KMH):
    """Tempo de viagem em minutos para uma distância em km e velocidade média."""
    return (distance_km / vel_kmh) * 60.0

# Converter distâncias para tempos
time_min = np.zeros_like(dist_km)
for i in range(len(dist_km)):
    for j in range(len(dist_km)):
        time_min[i, j] = travel_time_minutes_km(dist_km[i, j])

# Mapear IDs para índices
id_to_index = {node_id: idx for idx, node_id in enumerate(ids)}
index_to_id = {idx: node_id for node_id, idx in id_to_index.items()}

# -----------------------------
# Funções utilitárias
# -----------------------------
def route_service_time_minutes(route):
    """Tempo de serviço (descargas) somado nos nós internos (exclui depósito quando 0)."""
    t = 0.0
    for node_id in route:
        t += clientes[node_id]["descarga"] if node_id != 0 else 0.0
    return t

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
    # adicionar tempo de descarga para nós não-depósito
    total_time_min += route_service_time_minutes(route)
    return total_km, total_time_min

def route_load(route):
    """Soma de demandas (kg) dos clientes no route (exclui depósito 0)."""
    return sum(clientes[node]["demanda"] for node in route if node != 0)

# -----------------------------
# Heurísticas (mesmas do run_fast.py)
# -----------------------------
def sweep_routes():
    depot = clientes[0]
    angles = []
    for i in ids:
        if i == 0: continue
        dx = clientes[i]["lon"] - depot["lon"]
        dy = clientes[i]["lat"] - depot["lat"]
        ang = np.arctan2(dy, dx)
        angles.append((i, ang))
    angles.sort(key=lambda x: x[1])
    routes = []
    current_route = [0]
    current_load = 0
    for i, ang in angles:
        demand = clientes[i]["demanda"]
        tentative_route = current_route + [i, 0]
        load_if = current_load + demand
        if load_if <= CAPACIDADE:
            dist_km_val, time_min_val = route_distance_and_time(tentative_route)
            if time_min_val <= TEMPO_MAX_DIA_MIN:
                current_route = current_route + [i]
                current_load = load_if
            else:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0, i]
                current_load = demand
        else:
            current_route.append(0)
            routes.append(current_route)
            current_route = [0, i]
            current_load = demand
    current_route.append(0)
    routes.append(current_route)
    return routes

def nearest_neighbor_routes(farthest=False):
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
            tentative_route = route + [best, 0]
            tentative_load = load + clientes[best]["demanda"]
            if tentative_load <= CAPACIDADE:
                dist_km_val, time_min_val = route_distance_and_time(tentative_route)
                if time_min_val <= TEMPO_MAX_DIA_MIN:
                    route.append(best)
                    load = tentative_load
                    unserved.remove(best)
                    cur = best
                    continue
            break
        route.append(0)
        routes.append(route)
    return routes

def clarke_wright_routes():
    routes = {i: [0, i, 0] for i in ids if i != 0}
    route_loads = {i: clientes[i]["demanda"] for i in ids if i != 0}
    depot_idx = id_to_index[0]
    savings = []
    for i in ids:
        if i == 0: continue
        for j in ids:
            if j == 0 or j == i: continue
            s = dist_km[depot_idx, id_to_index[i]] + dist_km[depot_idx, id_to_index[j]] - dist_km[id_to_index[i], id_to_index[j]]
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
    final_routes = list(routes.values())
    return final_routes

# -----------------------------
# Gerar rotas com cada heurística
# -----------------------------
print("\nGerando rotas com heurísticas (usando dados CSV reais)...")

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
    
    # agendamento semanal
    days = [[] for _ in range(7)]
    days_time = [0.0 for _ in range(7)]
    sorted_summary = sorted(summary, key=lambda x: x["time_min"], reverse=True)
    for s in sorted_summary:
        placed = False
        for d in range(7):
            if days_time[d] + s["time_min"] <= TEMPO_MAX_DIA_MIN:
                days[d].append(s)
                days_time[d] += s["time_min"]
                placed = True
                break
        if not placed:
            min_idx = int(np.argmin(days_time))
            days[min_idx].append(s)
            days_time[min_idx] += s["time_min"]
            print("Aviso: uma rota não coube em nenhum dia sem exceder 6h; forçando atribuição.")
    
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
# Visualização com dados CSV
# -----------------------------
try:
    sol_name = "ClarkeWright"
    sol = results[sol_name]["summary"]
    print(f"\nPlotando rotas da solução {sol_name} (usando dados CSV)...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # plotar pontos
    for i in ids:
        lon = clientes[i]["lon"]; lat = clientes[i]["lat"]
        if i == 0:
            ax.scatter(lon, lat, s=150, marker='*', color='red', zorder=5, label='Depósito')
            ax.text(lon, lat + 0.005, " DEPÓSITO", fontsize=10, ha='center')
        else:
            ax.scatter(lon, lat, s=80, color='blue', zorder=5)
            ax.text(lon, lat + 0.003, f" {i}", fontsize=9, ha='center')
    
    # plotar rotas com linhas diretas
    colors = ["red","blue","green","orange","purple","brown","magenta","cyan"]
    for r_idx, r in enumerate(sol):
        route_nodes = r["route"]
        color = colors[r_idx % len(colors)]
        xs = [clientes[node]["lon"] for node in route_nodes]
        ys = [clientes[node]["lat"] for node in route_nodes]
        ax.plot(xs, ys, linewidth=3, alpha=0.7, color=color, zorder=3, 
                label=f'Rota {r_idx} ({r["load"]}kg, {r["time_min"]:.0f}min)')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Rotas ({sol_name}) - Brasília\nUsando distâncias reais dos dados CSV')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print("Plot falhou:", e)

print("\nExecução finalizada com dados CSV! Consulte as rotas e programação impressas acima.")
print("✓ Coordenadas: pontos.csv")
print("✓ Distâncias: distancias.csv (dados reais de roteamento)")