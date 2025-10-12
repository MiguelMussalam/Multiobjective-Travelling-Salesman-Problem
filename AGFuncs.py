import random
import numpy as np
from typing import Final

# Cidades que estarão no problema junto de suas coordernadas X e Y
CIDADES: Final[int] = {
    0: (0, 0),
    1: (2, 4),
    2: (5, 2),
    3: (6, 6),
    4: (8, 3)
} 

# Tempo de locomoção entre cidades (Horas)
tempos: Final[int] = (
    (None, 3,   1.5, 2,   9),
    (3,   None, 1,   15,  5),
    (2,   2,   None, 7,   2.5),
    (5,   6,   7,   None, 1),
    (1.5, 10,  3,   1,   None)
)

pedagio: Final[int] = (
    (None, 5,   2, 1,   3),
    (7,   None, 4,   7,  2),
    (2,   8,   None, 4,   8),
    (9,   6,   9,   None, 2),
    (2, 4,  7,   1,   None)
)

# Tamanho do cromossomo que sera igual ao tamanho de cidades do problema
TAM_CROMO: Final[int] = len(CIDADES)

# Tamanho da população que sera 2 vezes o tamanho do cromossomo 
TAM_POP: Final[int] = TAM_CROMO

# Declaração da população e qual tamanho ele tera
população = np.zeros((TAM_POP, TAM_CROMO))

# Fitness de cada cromossomo
nota_pop = np.zeros((TAM_POP, 3))

def iniciaPopulacao(populacao):
    # Cria o conjunto de população com valores aleatorios
    for i in range(len(populacao)):
        populacao[i] = np.random.permutation(TAM_CROMO)
  
# Calculo feito da distancia entre 2 cidades, cidadem1 e 2, depois eleva ambos ao quadrado pra retirar qual
def distancia(cidade1, cidade2):
    x1, y1 = CIDADES[(cidade1)]
    x2, y2 = CIDADES[(cidade2)]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Distancia total do percurso de cada cromossomo
def distanciaTotal(cromossomo):
    total = 0
    for i in range(TAM_CROMO - 1):
        total += distancia(cromossomo[i], cromossomo[i+1])
    # Volta para a cidade inicial
    total += distancia(cromossomo[-1], cromossomo[0])
    return total

def tempoTotal(cromossomo):
    total = 0
    for i in range(TAM_CROMO - 1):
        total += tempos[int(cromossomo[i])][int(cromossomo[i+1])]
    # Volta à cidade inicial
    total += tempos[int(cromossomo[-1])][int(cromossomo[0])]
    return total

def pedagioTotal(cromossomo):
    total = 0
    for i in range(TAM_CROMO - 1):
        total += pedagio[int(cromossomo[i])][int(cromossomo[i+1])]
    # Volta à cidade inicial
    total += pedagio[int(cromossomo[-1])][int(cromossomo[0])]
    return total

# Fitness de cada cromossomo, por enquanto apenas distancia
def fitness(populacao, nota_pop):
    pesos=(1, 1, 1)
    
    for i in range(len(populacao)):
        nota_pop[i, 0] = distanciaTotal(populacao[i])
        nota_pop[i, 1] = tempoTotal(populacao[i])
        nota_pop[i, 2] = pedagioTotal(populacao[i])
    
    # Normaliza os valores para evitar dominância de uma métrica
    if np.max(nota_pop[:, 0]) > 0:
        nota_pop[:, 0] = nota_pop[:, 0] / np.max(nota_pop[:, 0])
    if np.max(nota_pop[:, 1]) > 0:
        nota_pop[:, 1] = nota_pop[:, 1] / np.max(nota_pop[:, 1])
    if np.max(nota_pop[:, 2]) > 0:
        nota_pop[:, 2] = nota_pop[:, 2] / np.max(nota_pop[:, 2])
    
    # Combina os fitness com pesos
    fitness_combinado = (nota_pop[:, 0] * pesos[0] + 
                        nota_pop[:, 1] * pesos[1] + 
                        nota_pop[:, 2] * pesos[2])
    
    return fitness_combinado


def selecaoTorneio(população, fitness_combinado, k=3):
    if k > len(população):
        k = len(população)  # garante que k não seja maior que a população

    pais = np.zeros((2, TAM_CROMO))

    for p in range(2):
        candidatos = np.random.randint(0, len(população), k)
        # escolhe o melhor entre os candidatos baseado no fitness combinado
        melhor = candidatos[np.argmin(fitness_combinado[candidatos])]
        pais[p] = população[melhor]

    return pais

def crossoverOX(pai1, pai2):
    # -1 indica posição vazia
    filho = np.full(TAM_CROMO, -1)  

    # Escolhe aleatoriamente um segmento
    start, end = sorted(np.random.randint(0, TAM_CROMO, 2))
    
    # Copia segmento do pai1 para o filho
    filho[start:end+1] = pai1[start:end+1]
    
    # Preenche o restante com a ordem do pai2, sem repetir cidades
    pos_filho = (end+1) % TAM_CROMO
    for c in pai2:
        if c not in filho:
            filho[pos_filho] = c
            pos_filho = (pos_filho + 1) % TAM_CROMO
    
    return filho

def mutacao(cromossomo):
    taxa = 0.3
    if np.random.rand() < taxa:
        i, j = np.random.randint(0, TAM_CROMO, 2)
        cromossomo[i], cromossomo[j] = cromossomo[j], cromossomo[i]
    return cromossomo

def Elitismo(fitness_combinado):
    melhor_idx = np.argmin(fitness_combinado)
    
    return melhor_idx

def novaGeracao(populacao, fitness_combinado):
    
    nova_população = np.zeros((TAM_POP, TAM_CROMO))
    
    filho = np.zeros((2, TAM_CROMO))

    # Salva o melhor da geração anterior
    melhor_idx = np.argmin(fitness_combinado)
    nova_população[0] = populacao[melhor_idx]

    for i in range(1, TAM_POP):
        pais = selecaoTorneio(populacao, fitness_combinado)
        filho = crossoverOX(pais[0], pais[1])
        filho = mutacao(filho)  
        nova_população[i] = filho

    return nova_população
