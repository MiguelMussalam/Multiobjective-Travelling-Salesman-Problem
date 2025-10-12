import random
import numpy as np
from typing import Final

def inicia_populacao(populacao):
    # Cria o conjunto de população com valores aleatorios
    for i in range(TAM_POP):
        população[i] = np.random.permutation(TAM_CROMO)

# Calculo feito da distancia entre 2 cidades, cidadem1 e 2, depois eleva ambos ao quadrado pra retirar qual
def distancia(cidade1, cidade2):
    x1, y1 = CIDADES[(cidade1)]
    x2, y2 = CIDADES[(cidade2)]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Distancia total do percurso de cada cromossomo
def distancia_total(cromossomo):
    total = 0
    for i in range(TAM_CROMO - 1):
        total += distancia(cromossomo[i], cromossomo[i+1])
    # Volta para a cidade inicial
    total += distancia(cromossomo[-1], cromossomo[0])
    return total

def tempo_total(cromossomo):
    total = 0
    for i in range(TAM_CROMO - 1):
        total += TEMPO[int(cromossomo[i])][int(cromossomo[i+1])]
    # Volta à cidade inicial
    total += TEMPO[int(cromossomo[-1])][int(cromossomo[0])]
    return total

# Fitness de cada cromossomo, por enquanto apenas distancia
for i in range(TAM_POP):
    nota_pop[i, 0] = distancia_total(população[i])
    nota_pop[i, 1] = tempo_total(população[i]) 
    
def selecao_torneio(população, nota_pop, k=3):
    # k = tamanho do torneio (quantos indivíduos competem)
    # vetor criado com espaço vazio para 2 pais
    pais = np.zeros((2, TAM_CROMO))

    for p in range(2):  # selecionar dois pais
        # 0 tamanho minimo, TAM_POP tamanho maximo, k quantos cromossomos aleatórios gerar
        candidatos = np.random.randint(0, TAM_POP, k)
        # pega o cromossomo com menor fitness (distância total)
        melhor = candidatos[np.argmin(nota_pop[candidatos, 0])]
        pais[p] = população[melhor]
    
    return pais

def crossover_ox(pai1, pai2):
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

if __name__ == '__main__':
    # Declaração da população e qual tamanho ele tera
    população = np.zeros((TAM_POP, TAM_CROMO))
    inicia_populacao(população)

    # Nova população e qual tamanho ele tera
    nova_população = np.zeros((TAM_POP, TAM_CROMO))

    pais = np.zeros((2, TAM_CROMO))
    filhos = np.zeros((2, TAM_CROMO))

    # Fitness de cada cromossomo
    nota_pop = np.zeros((TAM_POP, 3))
    


    """     print("Populacao inicial:")
        print(população)
        for i in range(TAM_POP):
            print(f"Cromossomo {i}: {população[i]}  -> Distancia total: {distancia_total(população[i]):.3f}")

        for i in range(TAM_POP):
            print(f"Fitness Cromossomo {i}: {nota_pop[i,0]:.3f}")
        
        pais = selecao_torneio(população, nota_pop)
        print("Pais selecionados:")
        print(pais)

        filho = crossover_ox(pais[0], pais[1])
        print("Filho gerado pelo crossover:", filho)



        for i in range(len(CIDADES)):
            for j in range(len(CIDADES)):
                if(i != j):
                    print(f"Cidade {i} para Cidade {j} tempo: {TEMPO[i][j]:.3f}")
                    print(f"Cidade {i} para Cidade {j} Custo: {PRECO_PEDAGIO[i][j]:.3f}\n")

        for i in range(TAM_POP):
            print(f"Cromossomo {i}: {população[i]} -> Distancia: {nota_pop[i,0]:.3f} | Tempo: {nota_pop[i,1]:.3f}") """