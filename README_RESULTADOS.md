# 🚚 Roteirização de Veículos - Brasília DF

## 📋 Descrição do Projeto

Sistema completo de roteirização de veículos (VRP - Vehicle Routing Problem) para distribuição de produtos em Brasília-DF, utilizando dados reais de vias do OpenStreetMap (OSM) e implementando 4 algoritmos heurísticos clássicos.

---

## 🎯 Características do Problema

- **Depósito:** Carrefour STN, Brasília-DF
- **Clientes:** 9 pontos de entrega distribuídos pelo DF
- **Demanda Total:** 2.372 kg
- **Capacidade do Veículo:** 1.800 kg
- **Tempo Máximo por Rota:** 6 horas (360 minutos)
- **Velocidade Média:** 50 km/h
- **Rede Viária:** Dados reais do OpenStreetMap

---

## 🔧 Tecnologias Utilizadas

- **Python 3.x**
- **NetworkX** - Manipulação de grafos e cálculo de rotas
- **OSMnx** - Download e processamento de dados do OpenStreetMap
- **Matplotlib** - Visualizações estáticas
- **Folium** - Mapas interativos HTML
- **Pandas** - Manipulação de dados tabulares
- **NumPy** - Cálculos numéricos

---

## 🧮 Algoritmos Implementados

### 1. 🥇 Clarke & Wright (Savings Algorithm)
- **Distância Total:** 103.29 km
- **Princípio:** Calcula economias (savings) ao combinar rotas
- **Resultado:** Melhor desempenho geral

### 2. 🥈 Varredura (Sweep)
- **Distância Total:** 106.37 km (+3.0%)
- **Princípio:** Ordena clientes por ângulo polar e agrupa sequencialmente
- **Resultado:** Segundo melhor, muito competitivo

### 3. 🥉 Ponto Mais Distante (Farthest-First)
- **Distância Total:** 113.97 km (+10.3%)
- **Princípio:** Começa sempre com o cliente mais distante
- **Resultado:** Desempenho intermediário

### 4. Vizinho Mais Próximo (Nearest Neighbor)
- **Distância Total:** 116.46 km (+12.7%)
- **Princípio:** Escolhe sempre o cliente mais próximo não visitado
- **Resultado:** Mais simples, mas menos eficiente

---

## 📁 Estrutura de Arquivos

### Scripts Principais
```
run.py                      # Script principal - executa os 4 algoritmos
visualizar_comparacao.py    # Gera gráficos comparativos adicionais
download_osm.py             # Download dos dados do OpenStreetMap
exec1.py                    # Script auxiliar
```

### Dados de Entrada
```
data/pontos.csv            # Coordenadas dos pontos de entrega
brasilia.graphml           # Grafo da rede viária de Brasília (OSM)
```

### Arquivos de Saída (pasta `output/`)

#### 🗺️ Visualizações Principais
1. **comparacao_algoritmos.png** (1.2 MB)
   - Grade 2x2 comparando os 4 algoritmos visualmente
   - Mostra rotas traçadas sobre mapa simples
   - Estatísticas de cada algoritmo

2. **rotas_brasilia.png** (2.8 MB)
   - Mapa detalhado de alta resolução
   - Rotas do Clarke & Wright sobre rede viária real OSM
   - Visualização profissional para apresentações

3. **rotas_brasilia.html** (41 KB)
   - Mapa interativo do Folium
   - Permite zoom e navegação
   - Popups com informações de cada ponto

#### 📊 Gráficos Comparativos
4. **grafico_comparacao_barras.png** (280 KB)
   - Gráfico de barras comparando distâncias totais
   - Ranking visual dos algoritmos
   - Percentuais de diferença

5. **analise_detalhada_rotas.png** (365 KB)
   - Análise detalhada rota por rota
   - Compara distância, carga e tempo
   - Grid 2x2 para os 4 algoritmos

#### 📄 Documentação
6. **RESULTADOS_COMPARACAO.md** (5 KB)
   - Relatório completo em Markdown
   - Análise detalhada de cada algoritmo
   - Conclusões e recomendações

---

## 🚀 Como Executar

### 1. Instalar Dependências
```bash
pip install networkx osmnx matplotlib folium pandas numpy branca
```

### 2. Executar o Script Principal
```bash
python run.py
```

Este script irá:
- ✅ Carregar os dados dos clientes
- ✅ Baixar/carregar o grafo viário de Brasília
- ✅ Executar os 4 algoritmos de roteirização
- ✅ Gerar visualizações comparativas
- ✅ Salvar resultados na pasta `output/`

**Tempo de execução:** ~30-60 segundos (dependendo do hardware)

### 3. Gerar Gráficos Adicionais (Opcional)
```bash
python visualizar_comparacao.py
```

Gera:
- Gráficos de barras comparativos
- Análise detalhada das rotas

---

## 📊 Resultados Resumidos

| Algoritmo | Distância | Rotas | Melhor? |
|-----------|-----------|-------|---------|
| Clarke & Wright | 103.29 km | 2 | ⭐ Sim |
| Varredura | 106.37 km | 2 | 🥈 |
| Ponto Mais Distante | 113.97 km | 2 | 🥉 |
| Vizinho Mais Próximo | 116.46 km | 2 | - |

**Economia do Melhor vs Pior:** 13.17 km (11.3%)

---

## 📈 Destaques

### ✅ Pontos Fortes
- ✨ Usa dados reais de vias (OpenStreetMap)
- 🎯 Implementa 4 algoritmos clássicos
- 📊 Visualizações profissionais e interativas
- 🔍 Respeita todas as restrições (capacidade e tempo)
- 📝 Documentação completa

### 🎓 Aplicações
- Distribuição de produtos
- Logística urbana
- Planejamento de rotas
- Estudo de algoritmos heurísticos
- Trabalho acadêmico de Introdução à Logística

---


Trabalho de Introdução à Logística  
Data: Outubro 2025

---

## 📚 Referências

- Clarke, G., & Wright, J. W. (1964). Scheduling of vehicles from a central depot to a number of delivery points. *Operations Research*, 12(4), 568-581.
- Toth, P., & Vigo, D. (Eds.). (2014). *Vehicle routing: problems, methods, and applications*. Society for Industrial and Applied Mathematics.
- OpenStreetMap Contributors. (2024). *Planet dump*. Retrieved from https://www.openstreetmap.org

---

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos.

---

## 🔗 Links Úteis

- [OSMnx Documentation](https://osmnx.readthedocs.io/)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- [Folium Documentation](https://python-visualization.github.io/folium/)

---

**🎉 Projeto Completo e Funcional! 🎉**
