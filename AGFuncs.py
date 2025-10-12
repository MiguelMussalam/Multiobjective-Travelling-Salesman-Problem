import sys
import random
import numpy as np
from typing import Final
from tabulate import tabulate

# Cidades que estarão no problema junto de suas coordernadas X e Y
CIDADES: Final[int] = {
    0: (0, 0),
    1: (2, 4),
    2: (5, 2),
    3: (6, 6),
    4: (8, 3)
}

# Tempo de locomoção entre cidades (Horas)
TEMPO: Final = (
    (None, 3,   1.5, 2,   9),
    (3,   None, 1,   15,  5),
    (2,   2,   None, 7,   2.5),
    (5,   6,   7,   None, 1),
    (1.5, 10,  3,   1,   None)
)

# Quantidade de semáforos entre cidades
PRECO_PEDAGIO: Final = (
    (None, 10,   5, 7,   2),
    (7,   None, 3,   15,  2),
    (2,   11,   None, 4,   8),
    (10,   6,   9,   None, 11),
    (2, 4,  7,   1,   None)
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
    
    total += TEMPO[int(cromossomo[-1])][int(cromossomo[0])]
    return total

#       --NOTA POPULACAO--
# [i][0] -> Indice original na matriz 
# [i][1] -> Nota 
# [i][2] -> Distancia total 
# [i][3] -> Tempo total
# [i][4] -> preco pedágio
# [i][5] -> percentual de seleção

# Fitness de cada cromossomo, por enquanto apenas distancia
def fitness(populacao, nota_populacao):
    for i in range(len(populacao)):
        nota_populacao[i, 0] = i
        nota_populacao[i, 2] = distanciaTotal(populacao[i])
        nota_populacao[i, 3] = tempoTotal(populacao[i])
        nota_populacao[i, 4] = custoPedagio(populacao[i])
    
def selecaoTorneio(população, nota_populacao, k=3):
    # k = tamanho do torneio (quantos indivíduos competem)
    # vetor criado com espaço vazio para 2 pais
    pais = np.zeros((2, len(CIDADES)))

    for p in range(2):  # selecionar dois pais
        # 0 tamanho minimo, TAM_POP tamanho maximo, k quantos cromossomos aleatórios gerar
        candidatos = np.random.randint(0, len(população), k)
        # pega o cromossomo com menor fitness (distância total)
        melhor = candidatos[np.argmin(nota_populacao[candidatos, 0])]
        pais[p] = população[melhor]
    
    return pais

def crossoverOX(pai1, pai2):
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