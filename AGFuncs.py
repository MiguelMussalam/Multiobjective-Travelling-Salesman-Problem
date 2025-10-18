import sys
import random
import numpy as np
from typing import Final
from tabulate import tabulate
from numba import njit

# --- DEFINIÇÕES DO PROBLEMA (EXPANDIDO PARA 10 CIDADES) ---
CIDADES: Final[int] = {
    0: (0, 0),    # Cidade Original 0
    1: (2, 4),    # Cidade Original 1
    2: (5, 2),    # Cidade Original 2
    3: (6, 6),    # Cidade Original 3
    4: (8, 3),    # Cidade Original 4
    5: (10, 8),   # Nova Cidade 5
    6: (3, 9),    # Nova Cidade 6
    7: (12, 1),   # Nova Cidade 7
    8: (9, 11),   # Nova Cidade 8
    9: (1, 12)    # Nova Cidade 9
}

# Matriz de Tempo de locomoção entre cidades (Horas) - expandida para 10x10
TEMPO: Final = (
    #      0     1     2     3     4      5     6     7     8      9
    (None, 3,    1.5,  2,    9,     6,    4,    8.5,  11,    12.5), # 0
    (3,    None, 1,    15,   5,     7,    5,    10,   8,     10),   # 1
    (1.5,  1,    None, 7,    2.5,   4.5,  6,    5,    9,     11),   # 2
    (2,    15,   7,    None, 1,     8,    3.5,  12,   4,     9.5),  # 3
    (9,    5,    2.5,  1,    None,  2,    9,    4,    6,     13),   # 4
    (6,    7,    4.5,  8,    2,     None, 10,   3,    2.5,   14),   # 5
    (4,    5,    6,    3.5,  9,     10,   None, 14,   5,     4),    # 6
    (8.5,  10,   5,    12,   4,     3,    14,   None, 9,     16),   # 7
    (11,   8,    9,    4,    6,     2.5,  5,    9,    None,  3),    # 8
    (12.5, 10,   11,   9.5,  13,    14,   4,    16,   3,     None)   # 9
)

# Matriz de Custo de Pedágio entre cidades - expandida para 10x10
PRECO_PEDAGIO: Final = (
    #      0     1     2     3     4      5     6     7     8     9
    (None, 10,   5,    7,    2,     12,   8,    15,   20,   22),   # 0
    (10,   None, 3,    15,   2,     9,    6,    18,   14,   19),   # 1
    (5,    3,    None, 4,    8,     6,    10,   9,    16,   20),   # 2
    (7,    15,   4,    None, 11,    14,   5,    21,   7,    17),   # 3
    (2,    2,    8,    11,   None,  4,    13,   6,    10,   23),   # 4
    (12,   9,    6,    14,   4,     None, 16,   5,    8,    25),   # 5
    (8,    6,    10,   5,    13,    16,   None, 24,   9,    7),    # 6
    (15,   18,   9,    21,   6,     5,    24,   None, 17,   28),   # 7
    (20,   14,   16,   7,    10,    8,    9,    17,   None,  6),    # 8
    (22,   19,   20,   17,   23,    25,   7,    28,   6,    None)    # 9
)

@njit
def iniciaPopulacao(populacao):
    # Cria o conjunto de população com valores aleatorios
    for i in range(len(populacao)):
        populacao[i] = np.random.permutation(len(CIDADES))

# Calculo feito da distancia entre 2 cidades, cidadem1 e 2, depois eleva ambos ao quadrado pra retirar qual
@njit
def distancia(cidade1, cidade2):
    x1, y1 = CIDADES[(cidade1)]
    x2, y2 = CIDADES[(cidade2)]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Distancia total do percurso de cada cromossomo
@njit
def distanciaTotal(cromossomo):
    total = 0
    for i in range(len(cromossomo) - 1):
        total += distancia(cromossomo[i], cromossomo[i+1])
    # Volta para a cidade inicial
    total += distancia(cromossomo[-1], cromossomo[0])
    return total

@njit
def tempoTotal(cromossomo):
    total = 0
    for i in range(len(cromossomo) - 1):
        total += TEMPO[int(cromossomo[i])][int(cromossomo[i+1])]
    # Volta à cidade inicial
    total += TEMPO[int(cromossomo[-1])][int(cromossomo[0])]
    return total

@njit
def custoPedagio(cromossomo):
    total = 0
    for i in range(len(cromossomo) - 1):
        total += PRECO_PEDAGIO[int(cromossomo[i])][int(cromossomo[i+1])]
    
    total += PRECO_PEDAGIO[int(cromossomo[-1])][int(cromossomo[0])]
    return total

#       --NOTA POPULACAO--
# [i][0] -> Indice original na matriz 
# [i][1] -> Nota 
# [i][2] -> Distancia total 
# [i][3] -> Tempo total
# [i][4] -> preco pedágio
# [i][5] -> percentual de seleção

# Fitness de cada cromossomo, por enquanto apenas distancia
@njit
def calculoNotas(populacao, nota_populacao):
    for i in range(len(populacao)):
        nota_populacao[i, 0] = i
        nota_populacao[i, 2] = distanciaTotal(populacao[i])
        nota_populacao[i, 3] = tempoTotal(populacao[i])
        nota_populacao[i, 4] = custoPedagio(populacao[i])

@njit
def domina(indice_a, indice_b, nota_populacao):
    objetivos_a = nota_populacao[indice_a, 2:5]
    objetivos_b = nota_populacao[indice_b, 2:5]
    return np.all(objetivos_a <= objetivos_b) and np.any(objetivos_a < objetivos_b)

@njit
def calculoFronteDePareto(nota_populacao, TAM_POP):
    domination_counts = np.zeros(TAM_POP, dtype=int)
    dominated_solutions = [[] for _ in range(TAM_POP)]
    ranks = np.zeros(TAM_POP, dtype=int)
    fronts = [[]]

    for i in range(TAM_POP):
        for j in range(i + 1, TAM_POP):
            if domina(i, j, nota_populacao):
                dominated_solutions[i].append(j)
                domination_counts[j] += 1
            elif domina(j, i, nota_populacao):
                dominated_solutions[j].append(i)
                domination_counts[i] += 1
    
    for i in range(TAM_POP):
        if domination_counts[i] == 0:
            ranks[i] = 0
            fronts[0].append(i)
            
    rank_atual = 0
    # --- A CORREÇÃO ESTÁ NESTA LINHA ---
    while rank_atual < len(fronts):
        next_front = []
        for i in fronts[rank_atual]:
            for j in dominated_solutions[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    ranks[j] = rank_atual + 1
                    next_front.append(j)
        rank_atual += 1
        if len(next_front) > 0:
            fronts.append(next_front)
            
    return ranks, fronts

@njit
def calculate_crowding_metrics(nota_populacao, fronts):
    """
    Calcula a distância de aglomeração para cada indivíduo para manter a diversidade.
    """
    num_individuals = len(nota_populacao)
    crowding_metrics = np.zeros(num_individuals)
    
    objetivos = nota_populacao[:, 2:5] # Colunas de distancia, tempo, pedagio
    num_objectives = objetivos.shape[1]

    for front in fronts:
        if not front: continue # Pula se a fronteira estiver vazia
        
        front_objetivos = objetivos[front, :]
        
        for m in range(num_objectives):
            # Ordena a fronteira com base no objetivo atual
            sorted_indices = np.argsort(front_objetivos[:, m])
            sorted_front = np.array(front)[sorted_indices]
            
            # Atribui distância infinita aos extremos da fronteira
            crowding_metrics[sorted_front[0]] = np.inf
            crowding_metrics[sorted_front[-1]] = np.inf
            
            if len(sorted_front) > 2:
                min_obj = front_objetivos[sorted_indices[0], m]
                max_obj = front_objetivos[sorted_indices[-1], m]
                range_obj = max_obj - min_obj
                if range_obj == 0: continue # Evita divisão por zero

                # Calcula a distância para os pontos intermediários
                for i in range(1, len(sorted_front) - 1):
                    dist = objetivos[sorted_front[i+1], m] - objetivos[sorted_front[i-1], m]
                    crowding_metrics[sorted_front[i]] += dist / range_obj
                    
    return crowding_metrics

@njit
def selecao_NSGA2(ranks, crowding_metrics, k=2):
    """
    Seleciona o índice de um indivíduo usando Torneio Binário.
    Critérios: 1. Melhor Rank de Pareto, 2. Maior Crowding Distance (desempate).
    """
    pop_size = len(ranks)
    
    # Sorteia k competidores (k=2 para torneio binário)
    candidatos_indices = np.random.choice(pop_size, k, replace=False)
    
    idx1 = candidatos_indices[0]
    idx2 = candidatos_indices[1]
    
    # Critério 1: O indivíduo com o menor rank de Pareto vence
    if ranks[idx1] < ranks[idx2]:
        return idx1
    elif ranks[idx2] < ranks[idx1]:
        return idx2
    
    # Critério 2 (desempate): O indivíduo com a maior crowding distance vence
    if crowding_metrics[idx1] > crowding_metrics[idx2]:
        return idx1
    else:
        return idx2

@njit
def mutacao_swap(cromossomo, taxa_mutacao=0.1):
    """
    Aplica mutação trocando a posição de duas cidades.
    """
    if random.random() < taxa_mutacao:
        idx1, idx2 = np.random.choice(len(cromossomo), 2, replace=False)
        cromossomo[idx1], cromossomo[idx2] = cromossomo[idx2], cromossomo[idx1]
    return cromossomo

@njit
def crossoverOX(pai1, pai2, TAM_CROMO):
    # -1 indica posição vazia
    filho = np.full(TAM_CROMO, -1)  

    # Escolhe aleatoriamente um segmento
    start, end = sorted(np.random.randint(0, len(CIDADES), 2))
    
    # Copia segmento do pai1 para o filho
    filho[start:end+1] = pai1[start:end+1]
    
    # Preenche o restante com a ordem do pai2, sem repetir cidades
    pos_filho = (end+1) % len(CIDADES)
    for c in pai2:
        if c not in filho:
            filho[pos_filho] = c
            pos_filho = (pos_filho + 1) % len(CIDADES)
    
    return filho

@njit
def printPopulacao(populacao, notaPopulacao):
    headers = ["ID", "CROMOSSOMO", "DISTÂNCIA", "TEMPO", "PEDÁGIO"]
    table_data = []

    for i in range(len(populacao)):
        rota_str = " - ".join([str(int(cidade)) for cidade in populacao[i]])
        distancia = notaPopulacao[i][2]
        tempo = notaPopulacao[i][3]
        custo = notaPopulacao[i][4]
        table_data.append([i, rota_str, f"{distancia:.2f}KM", f"{tempo:.2f}H", f"{custo:.2f}R$"])

    print("\nPOPULAÇÃO ATUAL:")
    # A mágica acontece aqui!
    print(tabulate(table_data, headers=headers, tablefmt="grid"))