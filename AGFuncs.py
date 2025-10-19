import sys
import random
import numpy as np
from typing import Final
from tabulate import tabulate

CIDADES: Final[int] = {
    0: (5, 5),
    1: (10, 12),
    2: (15, 8),
    3: (8, 15),
    4: (90, 5),
    5: (85, 80), 
    6: (5, 80), 
    7: (45, 50),
    8: (50, 45),
    9: (20, 95)
}

TEMPO: Final = (
    #      0     1     2     3     4      5     6     7     8      9
    (None, 2.0,  3.0,  2.5,  15.0,  25.0,  18.0,  8.0,   9.0,   22.0), # 0
    (2.0,  None, 5.0,  8.0,  14.0,  23.0,  19.0,  7.0,   8.0,   20.0), # 1
    (3.0,  5.0,  None, 4.0,  12.0,  21.0,  22.0,  6.0,   7.0,   24.0), # 2
    (2.5,  8.0,  4.0,  None, 16.0,  24.0,  17.0,  7.5,   8.5,   18.0), # 3
    (15.0, 14.0, 12.0, 16.0, None,  10.0,  28.0,  9.0,   10.0,  30.0), # 4
    (25.0, 23.0, 21.0, 24.0, 10.0,  None,  15.0,  11.0,  10.0,  12.0), # 5
    (18.0, 19.0, 22.0, 17.0, 28.0,  15.0,  None,  12.0,  13.0,  3.0),  # 6
    (8.0,  7.0,  6.0,  7.5,  9.0,   11.0,  12.0,  None,  1.0,   14.0), # 7
    (9.0,  8.0,  7.0,  8.5,  10.0,  10.0,  13.0,  1.0,   None,  13.0), # 8
    (22.0, 20.0, 24.0, 18.0, 30.0,  12.0,  3.0,   14.0,  13.0,  None)  # 9
)

PRECO_PEDAGIO: Final = (
    #      0     1     2     3     4      5     6     7     8      9
    (None, 5,    8,    7,    80,    100,   10,    20,    22,    15),   # 0
    (5,    None, 15,   5,    70,    90,    25,    30,    35,    28),   # 1
    (8,    15,   None, 12,   60,    80,    30,    25,    28,    32),   # 2
    (7,    5,    12,   None, 75,    95,    20,    32,    38,    22),   # 3
    (80,   70,   60,   75,   None,  90,    5,     40,    45,    10),   # 4
    (100,  90,   80,   95,   90,    None,  110,   50,    48,    120),  # 5
    (10,   25,   30,   20,   5,     110,   None,  40,    42,    0),    # 6
    (20,   30,   25,   32,   40,    50,    40,    None,  0,     55),   # 7
    (22,   35,   28,   38,   45,    48,    42,    0,     None,  58),   # 8
    (15,   28,   32,   22,   10,    120,   0,     55,    58,    None)  # 9
)

def iniciaPopulacao(populacao):
    # Cria o conjunto de população com valores aleatorios
    for i in range(len(populacao)):
        populacao[i] = np.random.permutation(len(CIDADES))

# Calculo feito da distancia entre 2 cidades, cidadem1 e 2, depois eleva ambos ao quadrado pra retirar qual

def distancia(cidade1, cidade2):
    x1, y1 = CIDADES[(cidade1)]
    x2, y2 = CIDADES[(cidade2)]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Distancia total do percurso de cada cromossomo

def distanciaTotal(cromossomo):
    total = 0
    for i in range(len(cromossomo) - 1):
        total += distancia(cromossomo[i], cromossomo[i+1])
    # Volta para a cidade inicial
    total += distancia(cromossomo[-1], cromossomo[0])
    return total


def tempoTotal(cromossomo):
    total = 0
    for i in range(len(cromossomo) - 1):
        total += TEMPO[int(cromossomo[i])][int(cromossomo[i+1])]
    # Volta à cidade inicial
    total += TEMPO[int(cromossomo[-1])][int(cromossomo[0])]
    return total


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

def calculoNotas(populacao, nota_populacao):
    for i in range(len(populacao)):
        nota_populacao[i, 0] = i
        nota_populacao[i, 2] = distanciaTotal(populacao[i])
        nota_populacao[i, 3] = tempoTotal(populacao[i])
        nota_populacao[i, 4] = custoPedagio(populacao[i])


def domina(indice_a, indice_b, nota_populacao):
    objetivos_a = nota_populacao[indice_a, 2:5]
    objetivos_b = nota_populacao[indice_b, 2:5]
    return np.all(objetivos_a <= objetivos_b) and np.any(objetivos_a < objetivos_b)


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


def mutacao_swap(cromossomo, taxa_mutacao=0.1):
    """
    Aplica mutação trocando a posição de duas cidades.
    """
    if random.random() < taxa_mutacao:
        idx1, idx2 = np.random.choice(len(cromossomo), 2, replace=False)
        cromossomo[idx1], cromossomo[idx2] = cromossomo[idx2], cromossomo[idx1]
    return cromossomo


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