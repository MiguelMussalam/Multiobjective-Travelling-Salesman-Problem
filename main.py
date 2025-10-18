from AGFuncs import *
import numpy as np
from typing import Final
import matplotlib.pyplot as plt
import functools

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

# --- PARÂMETROS DO ALGORITMO ---
TAM_CROMO: Final[int] = len(CIDADES)
TAM_POP: Final[int] = TAM_CROMO * 10  # Aumentar a população é bom para diversidade
NUM_GERACOES: Final[int] = 1000
TAXA_MUTACAO: Final[float] = 0.03


# --- BLOCO PRINCIPAL DE EXECUÇÃO ---
if __name__ == '__main__':

    # --- 1. INICIALIZAÇÃO ---
    import csv
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm # Importa o módulo de mapas de cores

    populacao = np.zeros((TAM_POP, TAM_CROMO), dtype=int)
    nota_populacao = np.zeros((TAM_POP, 6))
    
    iniciaPopulacao(populacao)
    print("--- POPULAÇÃO INICIAL GERADA ---")
    
    # --- DADOS PARA O GRÁFICO ---
    # Agora vamos guardar a fronteira de CADA geração
    historico_plot = []

    # --- ABERTURA DO ARQUIVO CSV ---
    with open('historico_fronteiras.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Geracao', 'ID_Original', 'Cromossomo', 'Distancia_KM', 'Tempo_H', 'Pedagio_R$'])

        # --- 2. LOOP EVOLUTIVO ---
        for geracao in range(NUM_GERACOES):
            # (O print foi movido para o final do loop para não poluir a saída)
            
            # ETAPA A: AVALIAÇÃO
            calculoNotas(populacao, nota_populacao)
            
            # ETAPA B: RANQUEAMENTO
            ranks, fronts = calculoFronteDePareto(nota_populacao, TAM_POP)
            
            # ETAPA C: DIVERSIDADE
            crowding_distances = calculate_crowding_metrics(nota_populacao, fronts)
            
            if (geracao + 1) % 10 == 0 or geracao == 0: # Imprime o status a cada 10 gerações
                 print(f"\rProcessando Geração {geracao+1}/{NUM_GERACOES} | Membros na Fronteira: {len(fronts[0])}", end="")

            # --- SALVANDO DADOS DA GERAÇÃO ATUAL ---
            melhores_indices_geracao = fronts[0]
            
            # Salva os dados no CSV
            for idx in melhores_indices_geracao:
                rota_str = " - ".join([str(int(c)) for c in populacao[idx]])
                dist = nota_populacao[idx, 2]
                tempo = nota_populacao[idx, 3]
                custo = nota_populacao[idx, 4]
                csv_writer.writerow([geracao + 1, idx, rota_str, f"{dist:.2f}", f"{tempo:.2f}", f"{custo:.2f}"])

            # Salva a fronteira para o gráfico final
            historico_plot.append(nota_populacao[melhores_indices_geracao, 2:4])

            # ETAPA D: CRIAÇÃO DA NOVA GERAÇÃO
            nova_populacao = np.zeros_like(populacao)
            for i in range(TAM_POP):
                idx_pai1 = selecao_NSGA2(ranks, crowding_distances)
                idx_pai2 = selecao_NSGA2(ranks, crowding_distances)
                pai1 = populacao[idx_pai1]
                pai2 = populacao[idx_pai2]
                filho = crossoverOX(pai1, pai2, TAM_CROMO)
                filho = mutacao_swap(filho, TAXA_MUTACAO)
                nova_populacao[i] = filho
            
            # ETAPA E: SUBSTITUIÇÃO
            populacao = nova_populacao

    print("\n\nEvolução concluída.")
    # --- 3. RESULTADO FINAL (A última fronteira encontrada) ---
    fronteira_final = historico_plot[-1]
    # (A impressão da tabela final já é feita pelo gráfico e CSV)

    # --- 4. GERAÇÃO DO GRÁFICO DE CONVERGÊNCIA ---
    print("Gerando gráfico de convergência...")
    plt.figure(figsize=(14, 9))
    
    # Define o mapa de cores que vai do azul (início) ao vermelho (fim)
    cores = cm.viridis(np.linspace(0, 1, len(historico_plot)))

    # Plota a fronteira de cada geração com uma cor diferente e transparência
    for i, fronteira in enumerate(historico_plot):
        if len(fronteira) > 0:
            distancias = fronteira[:, 0]
            tempos = fronteira[:, 1]
            plt.scatter(distancias, tempos, color=cores[i], alpha=0.5, s=15)

    # Destaca a primeira e a última fronteira para clareza
    if len(historico_plot[0]) > 0:
        plt.scatter(historico_plot[0][:, 0], historico_plot[0][:, 1], color='cyan', s=100, marker='o', label='Geração 1', zorder=10)
    if len(fronteira_final) > 0:
        plt.scatter(fronteira_final[:, 0], fronteira_final[:, 1], color='red', s=100, marker='X', label=f'Geração Final ({NUM_GERACOES})', zorder=10)

    plt.title('Convergência da Fronteira de Pareto (NSGA-II)', fontsize=16)
    plt.xlabel('Distância Total (KM)', fontsize=12)
    plt.ylabel('Tempo Total (H)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig('convergencia_pareto_completa.png')
    plt.show()