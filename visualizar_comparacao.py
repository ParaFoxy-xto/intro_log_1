"""
visualizar_comparacao.py
Script para gerar gráficos de barras comparativos dos 4 algoritmos
"""

import matplotlib.pyplot as plt
import numpy as np

# Dados dos resultados
algoritmos = ['Clarke &\nWright', 'Vizinho Mais\nPróximo', 'Ponto Mais\nDistante', 'Varredura\n(Sweep)']
distancias = [103.29, 116.46, 113.97, 106.37]
num_rotas = [2, 2, 2, 2]

# Cores para cada algoritmo
cores = ['#2E7D32', '#C62828', '#F57C00', '#1565C0']

# Criar figura com 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: Distância Total
x_pos = np.arange(len(algoritmos))
bars1 = ax1.bar(x_pos, distancias, color=cores, alpha=0.8, edgecolor='black', linewidth=1.5)

# Adicionar valores nas barras
for i, (bar, dist) in enumerate(zip(bars1, distancias)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{dist:.2f} km',
             ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Adicionar percentual de diferença em relação ao melhor
    if i > 0:
        diff_pct = ((dist - distancias[0]) / distancias[0]) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                 f'+{diff_pct:.1f}%',
                 ha='center', va='center', fontsize=9, color='white', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

ax1.set_ylabel('Distância Total (km)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Algoritmo', fontsize=12, fontweight='bold')
ax1.set_title('Comparação de Distância Total por Algoritmo', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(algoritmos, fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(distancias) * 1.15)

# Destacar o melhor resultado
min_idx = distancias.index(min(distancias))
bars1[min_idx].set_edgecolor('gold')
bars1[min_idx].set_linewidth(3)
ax1.text(min_idx, distancias[min_idx] * 1.05, '⭐ MELHOR', 
         ha='center', fontsize=10, fontweight='bold', color='#FFD700')

# Gráfico 2: Comparação de eficiência (ranking invertido - menor é melhor)
ranking = [1, 4, 3, 2]  # Clarke, Nearest, Farthest, Sweep
bars2 = ax2.barh(algoritmos, distancias, color=cores, alpha=0.8, edgecolor='black', linewidth=1.5)

# Adicionar valores nas barras
for i, (bar, dist) in enumerate(zip(bars2, distancias)):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2.,
             f' {dist:.2f} km',
             ha='left', va='center', fontweight='bold', fontsize=11)

ax2.set_xlabel('Distância Total (km)', fontsize=12, fontweight='bold')
ax2.set_title('Ranking de Eficiência (Menor Distância)', fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_xlim(0, max(distancias) * 1.2)

# Adicionar ranking
for i, (alg, pos) in enumerate(zip(algoritmos, ranking)):
    if pos == 1:
        emoji = '🥇'
    elif pos == 2:
        emoji = '🥈'
    elif pos == 3:
        emoji = '🥉'
    else:
        emoji = f'{pos}º'
    ax2.text(5, i, emoji, ha='left', va='center', fontsize=14, fontweight='bold')

# Destacar o melhor resultado
bars2[min_idx].set_edgecolor('gold')
bars2[min_idx].set_linewidth(3)

plt.suptitle('Análise Comparativa de Heurísticas de Roteirização - Brasília', 
             fontsize=16, fontweight='bold', y=1.00)

plt.tight_layout()
plt.savefig('output/grafico_comparacao_barras.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico de barras salvo: output/grafico_comparacao_barras.png")
plt.show()

# Criar segundo gráfico: Detalhes de cada rota
print("\nGerando análise detalhada das rotas...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Dados das rotas para cada algoritmo
rotas_dados = {
    'Clarke & Wright': {
        'rotas': [
            {'nome': 'Rota 1', 'dist': 48.49, 'carga': 732, 'tempo': 338.2},
            {'nome': 'Rota 2', 'dist': 54.80, 'carga': 1640, 'tempo': 300.8}
        ],
        'cor': '#2E7D32'
    },
    'Vizinho Mais Próximo': {
        'rotas': [
            {'nome': 'Rota 1', 'dist': 48.80, 'carga': 692, 'tempo': 348.6},
            {'nome': 'Rota 2', 'dist': 67.66, 'carga': 1680, 'tempo': 306.2}
        ],
        'cor': '#C62828'
    },
    'Ponto Mais Distante': {
        'rotas': [
            {'nome': 'Rota 1', 'dist': 61.35, 'carga': 1670, 'tempo': 333.6},
            {'nome': 'Rota 2', 'dist': 52.62, 'carga': 702, 'tempo': 318.1}
        ],
        'cor': '#F57C00'
    },
    'Varredura (Sweep)': {
        'rotas': [
            {'nome': 'Rota 1', 'dist': 54.27, 'carga': 1640, 'tempo': 300.1},
            {'nome': 'Rota 2', 'dist': 52.10, 'carga': 732, 'tempo': 342.5}
        ],
        'cor': '#1565C0'
    }
}

for idx, (algo, dados) in enumerate(rotas_dados.items()):
    ax = axes[idx]
    
    # Dados das rotas
    rotas_nomes = [r['nome'] for r in dados['rotas']]
    distancias_rotas = [r['dist'] for r in dados['rotas']]
    cargas = [r['carga'] for r in dados['rotas']]
    tempos = [r['tempo'] for r in dados['rotas']]
    
    # Criar gráfico de barras agrupadas
    x = np.arange(len(rotas_nomes))
    width = 0.25
    
    bars1 = ax.bar(x - width, distancias_rotas, width, label='Distância (km)', 
                   color=dados['cor'], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, [c/10 for c in cargas], width, label='Carga (x100 kg)', 
                   color='orange', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, [t/10 for t in tempos], width, label='Tempo (x10 min)', 
                   color='purple', alpha=0.8, edgecolor='black')
    
    # Adicionar valores nas barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Valores', fontsize=10, fontweight='bold')
    ax.set_title(f'{algo}\nDistância Total: {sum(distancias_rotas):.2f} km', 
                fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(rotas_nomes)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle('Análise Detalhada das Rotas por Algoritmo', 
             fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('output/analise_detalhada_rotas.png', dpi=300, bbox_inches='tight')
print("✓ Análise detalhada salva: output/analise_detalhada_rotas.png")
plt.show()

print("\n✅ Todos os gráficos foram gerados com sucesso!")
print("Arquivos salvos em: output/")
