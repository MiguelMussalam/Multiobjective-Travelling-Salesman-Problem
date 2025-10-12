from AGFuncs import *
import numpy as np
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

# Tamanho do cromossomo que sera igual ao tamanho de cidades do problema
TAM_CROMO: Final[int] = len(CIDADES)

# Tamanho da população que sera X vezes o tamanho do cromossomo 
TAM_POP: Final[int] = TAM_CROMO * 2

pais = np.zeros((2, TAM_CROMO))
filhos = np.zeros((2, TAM_CROMO))


#       --NOTA POPULACAO--
# [i][0] -> Indice original na matriz 
# [i][1] -> Nota 
# [i][2] -> Distancia total 
# [i][3] -> Tempo total
# [i][4] -> preco pedágio
# [i][5] -> percentual de seleção
notaPopulacao = np.zeros((TAM_POP, 6))

if __name__ == '__main__':

    # Declaração da população e qual tamanho ele tera
    populacao = np.zeros((TAM_POP, TAM_CROMO))
    
    # Nova população e qual tamanho ele tera
    novaPopulacao = np.zeros((TAM_POP, TAM_CROMO))
    
    iniciaPopulacao(populacao)
    fitness(populacao, notaPopulacao)
    printPopulacao(populacao, notaPopulacao)

    pais = np.zeros((2, TAM_CROMO))
    filhos = np.zeros((2, TAM_CROMO))

    # Fitness de cada cromossomo
    notaPopulacao = np.zeros((TAM_POP, 3))