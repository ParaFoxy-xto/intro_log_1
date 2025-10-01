|        |                                                                                                                                  |     |                      |                             |     |
| ------ | -------------------------------------------------------------------------------------------------------------------------------- | --- | -------------------- | --------------------------- | --- |
|        | Uma distribuidora de pães, bolos e biscoitos entrega esses produtos às                                                           |     |                      |                             |     |
|        | padarias e panificadoras localizadas no DF, conforme dados a seguir relacionados                                                 |     |                      |                             |     |
|        |                                                                                                                                  |     |                      |                             |     |
|        | DADOS DOS CLIENTES                                                                                                               |     |                      |                             |     |
|        |                                                                                                                                  |     |                      |                             |     |
|        | Endereço do Cliente                                                                                                              |     | Kg de produto/semana | Tempo de Descarga (minutos) |     |
| Número |                                                                                                                                  |     |                      |                             |     |
| 1      | CLS 307                                                                                                                          |     | 160                  | 50                          |     |
| 2      | CLS 114                                                                                                                          |     | 170                  | 60                          |     |
| 3      | CLN 110                                                                                                                          |     | 22                   | 70                          |     |
| 4      | SOF próximo Água Mineral                                                                                                         |     | 300                  | 85                          |     |
| 5      | SHIS QI 17 COMERCIAL                                                                                                             |     | 250                  | 45                          |     |
| 6      | CLSW 103                                                                                                                         |     | 90                   | 65                          |     |
| 7      | VARJÃO - entrada                                                                                                                 |     | 130                  | 55                          |     |
| 8      | ÁGUAS CLARAS - shopping                                                                                                          |     | 350                  | 40                          |     |
| 9      | TAGUATINGA PISTAO SUL                                                                                                            |     | 900                  | 45                          |     |
|        | Sede do Depósito - STN - ao lado do Carrefour                                                                                    |     | 2372                 | 515                         |     |
|        |                                                                                                                                  |     |                      |                             |     |
|        | OUTROS DADOS E INTRUÇÕES                                                                                                         |     |                      |                             |     |
|        | a )Frota tem um veículo com capacidade de carga de 1800 kg                                                                       |     |                      |                             |     |
|        | b) As distâncias reais devem ser buscadas no Google Maps ou similar                                                              |     |                      |                             |     |
|        | c) Tempo máximo de ciclo no dia = 6 horas                                                                                        |     |                      |                             |     |
|        | d) Velocidade Média = 50 km/hora                                                                                                 |     |                      |                             |     |
|        | e) Deve ser apresentada a programação de entregas, na semana, optando-se por um ou mais dias de entregas                         |     |                      |                             |     |
|        | f) O trabalho deve ser entregue digitalizado (será aberta uma tarefa oportunamente no Moodle/Teams).                             |     |                      |                             |     |
|        |                                                                                                                                  |     |                      |                             |     |
|        | Deve ser feita a roteirização por Clarke e Wrigth e para os métodos do ponto mais próximo, do ponto mais distante e da varredura |     |                      |                             |     |

---

# Solução do Exercício gemini

Aqui está a solução passo a passo para o problema de roteirização, utilizando os quatro métodos solicitados.

**Resumo dos Dados:**
- **Clientes:** 9
- **Veículo:** 1, com capacidade de 1800 kg.
- **Tempo Máximo de Rota:** 360 minutos.
- **Distâncias:** Calculadas em linha reta com fator de correção de 1.4.

**Nomenclatura dos Clientes:**
- **1:** CLS 307
- **2:** CLS 114
- **3:** CLN 110
- **4:** SOF Sul
- **5:** SHIS QI 17
- **6:** CLSW 103
- **7:** Varjão
- **8:** Águas Claras
- **9:** Taguatinga

---

## Resultados da Roteirização

A seguir são apresentadas as rotas geradas por cada um dos métodos.

### Método de Clarke e Wright (Savings)
- **Rota 1:** Depósito -> 3 -> 1 -> 5 -> 2 -> 6 -> Depósito
  - **Carga:** 692 kg (38.44%)
  - **Duração:** 342.85 min (95.24%)
  - **Distância:** 44.05 km
- **Rota 2:** Depósito -> 7 -> 9 -> 4 -> 8 -> Depósito
  - **Carga:** 1680 kg (93.33%)
  - **Duração:** 311.08 min (86.41%)
  - **Distância:** 71.73 km
- **Distância Total (todas as rotas):** 115.78 km

### Método do Ponto Mais Próximo
- **Rota 1:** Depósito -> 7 -> 3 -> 1 -> 6 -> 2 -> 5 -> Depósito
  - **Carga:** 822 kg (45.67%)
  - **Duração:** 417.79 min (116.05%)  **<-- ATENÇÃO: Tempo excedido!**
  - **Distância:** 60.66 km
- **Rota 2:** Depósito -> 8 -> 4 -> 9 -> Depósito
  - **Carga:** 1550 kg (86.11%)
  - **Duração:** 241.89 min (67.19%)
  - **Distância:** 59.91 km
- **Distância Total (todas as rotas):** 120.56 km

*Observação: A heurística do Ponto Mais Próximo, por ser muito "gananciosa", nem sempre produz soluções que respeitam todas as restrições, como visto na Rota 1.*

### Método de Inserção Mais Distante
- **Rota 1:** Depósito -> 7 -> 5 -> 4 -> 6 -> Depósito
  - **Carga:** 770 kg (42.78%)
  - **Duração:** 352.29 min (97.86%)
  - **Distância:** 85.24 km
- **Rota 2:** Depósito -> 3 -> 1 -> 2 -> 8 -> 9 -> Depósito
  - **Carga:** 1602 kg (89.00%)
  - **Duração:** 344.14 min (95.59%)
  - **Distância:** 65.95 km
- **Distância Total (todas as rotas):** 151.19 km

### Método de Varredura (Sweep)
- **Rota 1:** Depósito -> 2 -> 6 -> 4 -> 8 -> Depósito
  - **Carga:** 910 kg (50.56%)
  - **Duração:** 329.59 min (91.55%)
  - **Distância:** 66.33 km
- **Rota 2:** Depósito -> 9 -> 7 -> 3 -> 1 -> Depósito
  - **Carga:** 1212 kg (67.33%)
  - **Duração:** 322.79 min (89.66%)
  - **Distância:** 85.66 km
- **Rota 3:** Depósito -> 5 -> Depósito
  - **Carga:** 250 kg (13.89%)
  - **Duração:** 94.00 min (26.11%)
  - **Distância:** 40.84 km
- **Distância Total (todas as rotas):** 192.82 km

---
## Conclusão

O método de **Clarke e Wright** foi o que apresentou a menor distância total (115.78 km) e, portanto, a solução mais eficiente entre as heurísticas testadas, respeitando todas as restrições do problema.

---
# Minha Solução 

Esta é a sua solução, formatada para melhor legibilidade.

### Coordenadas Utilizadas
- **Depósito (Carrefour STN):** Lat: -15.7366, Lon: -47.90732
- **CLS 307:** Lat: -15.8122664, Lon: -47.9013959
- **CLS 114:** Lat: -15.8268977, Lon: -47.9191361
- **CLN 110:** Lat: -15.7743127, Lon: -47.88647
- **SOF (próx. Água Mineral):** Lat: -15.738056, Lon: -47.926667 (Parque Água Mineral)
- **SHIS QI 17 (Lago Sul):** Lat: -15.845, Lon: -47.862 (Aproximado)
- **CLSW 103:** Lat: -15.8010635, Lon: -47.9248713
- **Varjão (entrada):** Lat: -15.70972, Lon: -47.87889
- **Águas Claras (shopping):** Lat: -15.84028, Lon: -48.02778 (Av. das Araucárias)
- **Taguatinga Pistão Sul:** Lat: -15.851861, Lon: -48.041972 (Centro/Pistão Sul)

---
### Análise dos Agrupamentos de Clientes
As heurísticas agruparam os clientes em dois conjuntos principais:

- **Grupo 1 (Rota A):**
  - **Clientes:** CLN 110, CLS 114, SHIS QI 17, CLS 307, CLSW 103
  - **Carga Total:** 692 kg
  - **Tempo de Serviço (Descargas):** 290 min

- **Grupo 2 (Rota B):**
  - **Clientes:** SOF, Águas Claras, Taguatinga Pistão Sul, Varjão
  - **Carga Total:** 1680 kg
  - **Tempo de Serviço (Descargas):** 225 min

---
## Resultados por Heurística

### 1. Clarke & Wright (Savings)
- **Rota A:**
  - **Sequência:** Depósito → CLN 110 → CLS 114 → SHIS_QI17 → CLS 307 → CLSW 103 → Depósito
  - **Detalhes:** Carga: 692 kg, Duração Total: **~330.6 min**, Distância: ~33.8 km
- **Rota B:**
  - **Sequência:** Depósito → SOF → Águas Claras → Taguatinga → Varjão → Depósito
  - **Detalhes:** Carga: 1680 kg, Duração Total: **~282.1 min**, Distância: ~47.6 km

### 2. Vizinho Mais Próximo (Nearest Neighbor)
- **Rota A:**
  - **Sequência:** Depósito → CLN 110 → CLS 114 → SHIS QI 17 → CLS 307 → CLSW 103 → Depósito
  - **Detalhes:** Carga: 692 kg, Duração Total: **~331 min**
- **Rota B:**
  - **Sequência:** Depósito → Varjão → Taguatinga → Águas Claras → SOF → Depósito
  - **Detalhes:** Carga: 1680 kg, Duração Total: **~282 min**

### 3. Ponto Mais Distante (Farthest-First)
- **Rota B:**
  - **Sequência:** Depósito → Taguatinga → Águas Claras → SOF → Varjão → Depósito
  - **Detalhes:** Carga: 1680 kg, Duração Total: **~282 min**
- **Rota A:**
  - **Sequência:** Depósito → SHIS QI 17 → CLS 114 → CLS 307 → CLSW 103 → CLN 110 → Depósito
  - **Detalhes:** Carga: 692 kg, Duração Total: **~331 min**

### 4. Varredura (Sweep)
- **Rota A:**
  - **Sequência:** Depósito → CLN 110 → CLS 114 → SHIS QI 17 → CLS 307 → CLSW 103 → Depósito
  - **Detalhes:** Carga: 692 kg, Duração Total: **~331 min**
- **Rota B:**
  - **Sequência:** Depósito → SOF → Águas Claras → Taguatinga → Varjão → Depósito
  - **Detalhes:** Carga: 1680 kg, Duração Total: **~282 min**