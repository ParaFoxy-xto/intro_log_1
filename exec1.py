import folium
import pandas as pd
import os

# Tentar carregar dados do CSV primeiro
def load_points_from_csv():
    if os.path.exists('data/pontos.csv'):
        try:
            df = pd.read_csv('data/pontos.csv')
            pontos = []
            for idx, row in df.iterrows():
                tipo = "deposito" if "Depósito" in row['Nome'] else "ponto"
                pontos.append({
                    "nome": row['Nome'],
                    "lat": row['Latitude'],
                    "lon": row['Longitude'],
                    "tipo": tipo
                })
            print(f"✓ {len(pontos)} pontos carregados do CSV")
            return pontos
        except Exception as e:
            print(f"Erro ao carregar CSV: {e}")
            return None
    return None

# Tentar carregar do CSV, senão usar dados hardcoded
pontos_csv = load_points_from_csv()

if pontos_csv:
    pontos = pontos_csv
else:
    # Coordenadas dos pontos (dados originais como fallback)
    pontos = [
        {"nome": "Depósito (Carrefour STN)", "lat": -15.7366, "lon": -47.90732, "tipo": "deposito"},
        {"nome": "CLS 307", "lat": -15.8122664, "lon": -47.9013959, "tipo": "ponto"},
        {"nome": "CLS 114", "lat": -15.8268977, "lon": -47.9191361, "tipo": "ponto"},
        {"nome": "CLN 110", "lat": -15.7743127, "lon": -47.88647, "tipo": "ponto"},
        {"nome": "SOF (Água Mineral)", "lat": -15.738056, "lon": -47.926667, "tipo": "ponto"},
        {"nome": "SHIS QI 17 (Lago Sul)", "lat": -15.845, "lon": -47.862, "tipo": "ponto"},
        {"nome": "CLSW 103", "lat": -15.8010635, "lon": -47.9248713, "tipo": "ponto"},
        {"nome": "Varjão (entrada)", "lat": -15.70972, "lon": -47.87889, "tipo": "ponto"},
        {"nome": "Águas Claras (shopping)", "lat": -15.84028, "lon": -48.02778, "tipo": "ponto"},
        {"nome": "Taguatinga Pistão Sul", "lat": -15.851861, "lon": -48.041972, "tipo": "ponto"},
    ]

# Centralizar o mapa em Brasília
mapa = folium.Map(location=[-15.78, -47.93], zoom_start=12)

# Adicionar pontos ao mapa
for p in pontos:
    cor = "red" if p["tipo"] == "deposito" else "blue"
    folium.Marker(
        location=[p["lat"], p["lon"]],
        popup=p["nome"],
        tooltip=p["nome"],
        icon=folium.Icon(color=cor, icon="info-sign")
    ).add_to(mapa)

# Salvar em HTML
mapa.save("mapa_brasilia.html")

print(f"Mapa gerado com {len(pontos)} pontos! Abra o arquivo 'mapa_brasilia.html' no navegador.")
if pontos_csv:
    print("✓ Usando coordenadas atualizadas do CSV")
else:
    print("✓ Usando coordenadas hardcoded (coloque pontos.csv em /data para usar dados atualizados)")
