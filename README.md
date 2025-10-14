```markdown
# Roteirização de Entregas em Brasília (VRP) - OSMnx

Este projeto implementa a **roteirização de entregas** para clientes em Brasília utilizando **infraestrutura viária real** via **OpenStreetMap** (OSMnx) e heurísticas clássicas de **Problema de Roteirização de Veículos (VRP)**.

---

## Funcionalidades

- Baixa o grafo viário de Brasília usando **OSMnx**.
- Mapeia clientes e depósito aos nós mais próximos do grafo.
- Calcula **distâncias reais** (km) e **tempos de viagem** (minutos) via menor caminho no grafo.
- Considera:
  - Capacidade do veículo (1800 kg)
  - Tempo máximo de rota por dia (6 horas)
  - Tempo de descarga por cliente
  - Velocidade média de deslocamento (50 km/h)
- Implementa 4 heurísticas de roteirização:
  1. **Sweep (varredura por ângulo)**
  2. **Vizinho mais próximo**
  3. **Vizinho mais distante**
  4. **Clarke & Wright Savings**
- Gera **programação semanal** de entregas respeitando limite diário de 6 horas.
- Plota o grafo de Brasília com rotas sobrepostas (ex.: Clarke & Wright).

---

## Estrutura do Projeto

```

vrp_brasilia_osmnx/
│
├─ vrp_brasilia_osmnx.py      # Script principal
├─ README.md                  # Este arquivo
└─ requirements.txt           # Dependências Python

````

---

## Instalação

1. Clonar o repositório ou copiar os arquivos para seu computador.
2. Instalar as dependências:

```bash
pip install osmnx networkx matplotlib numpy
````

> Observação: OSMnx pode exigir dependências extras dependendo do seu sistema. Verifique a documentação oficial se houver erro.

---

## Uso

1. Abrir o terminal na pasta do projeto.
2. Executar o script Python:

```bash
python vrp_brasilia_osmnx.py
```

3. O script irá:

   * Baixar e construir o grafo viário de Brasília
   * Mapear os clientes ao grafo
   * Calcular rotas usando as heurísticas
   * Imprimir resumo de cada rota, incluindo:

     * Rota completa (nomes dos clientes)
     * Carga total (kg)
     * Distância total (km)
     * Tempo total (minutos)
   * Gerar uma **programação semanal** de rotas
   * Plotar o grafo com as rotas sobrepostas (opcional)

---

## Parâmetros Configuráveis

* `CAPACIDADE`: Capacidade máxima do veículo em kg (default 1800).
* `VEL_KMH`: Velocidade média de deslocamento em km/h (default 50).
* `TEMPO_MAX_DIA_MIN`: Tempo máximo por ciclo/dia em minutos (default 360 min = 6h).
* `clientes`: Dicionário com informações de clientes (ID, nome, coordenadas, demanda, tempo de descarga).

---

## Heurísticas Implementadas

1. **Sweep (varredura por ângulo)**

   * Ordena clientes pelo ângulo polar em relação ao depósito.
   * Agrupa clientes em rotas respeitando capacidade e tempo máximo.

2. **Vizinho mais próximo (Nearest Neighbor)**

   * Constrói rotas greedy selecionando o cliente mais próximo do último atendido.

3. **Vizinho mais distante (Farthest Neighbor)**

   * Constrói rotas greedy selecionando o cliente mais distante do último atendido.

4. **Clarke & Wright Savings**

   * Calcula “economias” ao combinar rotas individuais.
   * Concatena rotas se respeitarem capacidade e tempo máximo.

---

## Saída

* Rotas por heurística com:

  * Sequência de clientes
  * Distância total (km)
  * Tempo total (min)
  * Carga transportada (kg)
* Programação semanal indicando quais rotas acontecem em cada dia.
* Gráfico do grafo de Brasília com rotas sobrepostas.

---

## Observações

* A matriz de distâncias é calculada usando **menor caminho no grafo OSM** (infra real).
* Caso não exista caminho entre dois clientes no grafo, é usado fallback **Haversine**.
* A programação semanal tenta respeitar o limite diário de 6h, mas se uma rota exceder, ela será atribuída ao dia com menor carga e será exibido aviso.
* O código é modular e permite fácil substituição de heurísticas ou inclusão de otimizações adicionais (ex.: 2-opt, meta-heurísticas, ACO).

---

```
