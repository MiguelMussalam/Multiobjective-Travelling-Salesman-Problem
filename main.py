from AGFuncs import *
import numpy as np
from typing import Final
import matplotlib.pyplot as plt
import functools
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# --- PARÂMETROS DO ALGORITMO ---
TAM_CROMO: Final[int] = len(CIDADES)
TAM_POP: Final[int] = TAM_CROMO * 10
NUM_GERACOES: Final[int] = 500
TAXA_MUTACAO: Final[float] = 0.03
contador_estagnacao = 0
GERACOES_PARA_PARAR = 50 

if __name__ == '__main__':

    # --- 1. INICIALIZAÇÃO ---
    populacao = np.zeros((TAM_POP, TAM_CROMO), dtype=int)
    nota_populacao = np.zeros((TAM_POP, 6))
    iniciaPopulacao(populacao)
    print("--- POPULAÇÃO INICIAL GERADA ---")
    
    # ### Pré-avalia a população inicial antes de entrar no loop
    calculoNotas(populacao, nota_populacao)
    
    historico_plot = []

    with open('historico_fronteiras.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Geracao', 'ID_Original', 'Cromossomo', 'Distancia_KM', 'Tempo_H', 'Pedagio_R$'])

        for geracao in range(NUM_GERACOES):
            # ### PASSO 1: CRIAÇÃO DE FILHOS (Q) A PARTIR DA POPULAÇÃO ATUAL (P)
            
            # Ordena a população de pais (P) para usar na seleção por torneio
            ranks, fronts = calculoFronteDePareto(nota_populacao, TAM_POP)
            crowding_distances = calculate_crowding_metrics(nota_populacao, fronts)
            if fronts[0] == 100:
                break
            if (geracao + 1) % 10 == 0 or geracao == 0:
                 print(f"\rProcessando Geração {geracao+1}/{NUM_GERACOES} | Membros na Fronteira: {len(fronts[0])}", end="")

            melhores_indices_geracao = fronts[0]
            for idx in melhores_indices_geracao:
                rota_str = " - ".join([str(int(c)) for c in populacao[idx]])
                dist = nota_populacao[idx, 2]
                tempo = nota_populacao[idx, 3]
                custo = nota_populacao[idx, 4]
                csv_writer.writerow([geracao + 1, idx, rota_str, f"{dist:.2f}", f"{tempo:.2f}", f"{custo:.2f}"])
            historico_plot.append(nota_populacao[melhores_indices_geracao, 2:5])

            # Cria a população de filhos (Q)
            populacao_filhos = np.zeros_like(populacao)
            for i in range(TAM_POP):
                idx_pai1 = selecao_NSGA2(ranks, crowding_distances)
                idx_pai2 = selecao_NSGA2(ranks, crowding_distances)
                pai1 = populacao[idx_pai1]
                pai2 = populacao[idx_pai2]
                filho = crossoverOX(pai1, pai2, TAM_CROMO)
                filho = mutacao_swap(filho, TAXA_MUTACAO)
                populacao_filhos[i] = filho
            
            # ### PASSO 2: COMBINAÇÃO DE PAIS E FILHOS (R = P U Q)
            # ### AQUI ACONTECE A "REALIMENTAÇÃO". Pais e filhos são colocados juntos.
            populacao_combinada = np.vstack([populacao, populacao_filhos])
            
            # Avalia os objetivos para a população combinada de tamanho 2*N
            nota_populacao_combinada = np.zeros((TAM_POP * 2, 6))
            calculoNotas(populacao_combinada, nota_populacao_combinada)

            # ### PASSO 3: ORDENAÇÃO DA POPULAÇÃO COMBINADA (R)
            ranks_combinados, fronts_combinados = calculoFronteDePareto(nota_populacao_combinada, TAM_POP * 2)
            crowding_combinados = calculate_crowding_metrics(nota_populacao_combinada, fronts_combinados)
            if len(fronts[0]) == TAM_POP:
                contador_estagnacao += 1
            else:
                contador_estagnacao = 0 # Reseta o contador se a condição for quebrada

            if contador_estagnacao >= GERACOES_PARA_PARAR:
                print(f"\nParando na Geração {geracao+1} devido à estagnação da fronteira por {GERACOES_PARA_PARAR} gerações.")
                break
            # ### PASSO 4: SELEÇÃO DOS SOBREVIVENTES PARA A PRÓXIMA GERAÇÃO
            proxima_populacao = np.zeros_like(populacao)
            proxima_nota_populacao = np.zeros_like(nota_populacao)
            
            prox_idx_livre = 0
            front_num = 0
            
            # Preenche a nova população com as fronteiras, em ordem, até o limite de TAM_POP
            while prox_idx_livre < TAM_POP:
                indices_da_fronteira_atual = fronts_combinados[front_num]
                
                # Se a fronteira inteira não couber, precisaremos selecionar os mais diversos
                if prox_idx_livre + len(indices_da_fronteira_atual) > TAM_POP:
                    break 
                
                # Adiciona todos os indivíduos da fronteira atual
                for idx in indices_da_fronteira_atual:
                    proxima_populacao[prox_idx_livre] = populacao_combinada[idx]
                    proxima_nota_populacao[prox_idx_livre] = nota_populacao_combinada[idx]
                    prox_idx_livre += 1
                front_num += 1
            

            # Se ainda houver espaço, preenche com os melhores da próxima fronteira (a que não coube)
            if prox_idx_livre < TAM_POP:
                ultima_fronteira = fronts_combinados[front_num]
                
                # Obtém a distância de aglomeração apenas dos indivíduos dessa última fronteira
                crowding_da_fronteira = crowding_combinados[ultima_fronteira]
                
                # Ordena os índices da fronteira pela maior distância (descendente)
                indices_ordenados = np.argsort(-crowding_da_fronteira) # Negativo para ordem decrescente
                
                # Preenche o restante da população com os indivíduos mais diversos da última fronteira
                for i in range(TAM_POP - prox_idx_livre):
                    idx_original_na_comb = ultima_fronteira[indices_ordenados[i]]
                    proxima_populacao[prox_idx_livre + i] = populacao_combinada[idx_original_na_comb]
                    proxima_nota_populacao[prox_idx_livre + i] = nota_populacao_combinada[idx_original_na_comb]

            # ### PASSO 5: ATUALIZAÇÃO FINAL
            # A população de sobreviventes se torna a população de pais para a próxima geração
            populacao = proxima_populacao
            nota_populacao = proxima_nota_populacao

        print("\n--- MELHORES SOLUÇÕES ENCONTRADAS (Fronteira de Pareto da Geração Final) ---")

        # As variáveis `populacao` e `nota_populacao` já contêm os dados da última geração.
        # Apenas precisamos extrair a primeira fronteira (rank 0).
        final_ranks, final_fronts = calculoFronteDePareto(nota_populacao, TAM_POP)
        melhores_indices_finais = final_fronts[0]

        # Prepara os dados para a exibição em tabela
        headers = ["ID na Pop. Final", "Rota (Cromossomo)", "Distância (KM)", "Tempo (H)", "Pedágio (R$)"]
        table_data = []

        for idx in melhores_indices_finais:
                # Cria uma string mais clara para a rota, mostrando o retorno à origem
                cromossomo = populacao[idx]
                rota_str = " -> ".join(map(str, cromossomo)) + f" -> {cromossomo[0]}"

                # Pega os valores dos objetivos
                dist = nota_populacao[idx, 2]
                tempo = nota_populacao[idx, 3]
                custo = nota_populacao[idx, 4]

                table_data.append([
                        idx,
                        rota_str,
                        f"{dist:.2f}",
                        f"{tempo:.2f}",
                        f"{custo:.2f}"
                ])

        # Imprime a tabela formatada
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        #print(f'Dentre os 3 objetivos, temos na população final, o melhor para cada um deles:\n menor distância: {historico_plot}')

    # --- Seu código para gerar o gráfico permanece exatamente o mesmo ---
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    cores = cm.viridis(np.linspace(0, 1, NUM_GERACOES))
    for i, fronteira in enumerate(historico_plot):
        if len(fronteira) > 0:
            ax.scatter(fronteira[:, 0], fronteira[:, 1], fronteira[:, 2], color=cores[i], alpha=0.4, s=15)
    fronteira_inicial = historico_plot[0]
    fronteira_final = historico_plot[-1]
    if len(fronteira_inicial) > 0:
        ax.scatter(fronteira_inicial[:, 0], fronteira_inicial[:, 1], fronteira_inicial[:, 2], 
                   color='cyan', s=150, marker='o', label='Geração 1', depthshade=False, zorder=10)
    if len(fronteira_final) > 0:
        ax.scatter(fronteira_final[:, 0], fronteira_final[:, 1], fronteira_final[:, 2], 
                   color='red', s=150, marker='X', label=f'Geração Final ({NUM_GERACOES})', depthshade=False, zorder=10)
    ax.set_title('Convergência da Fronteira de Pareto 3D (NSGA-II)', fontsize=16)
    ax.set_xlabel('Distância Total (KM)', fontsize=12)
    ax.set_ylabel('Tempo Total (H)', fontsize=12)
    ax.set_zlabel('Custo Pedágio (R$)', fontsize=12)
    ax.legend()
    ax.view_init(elev=20, azim=45)
    plt.savefig('convergencia_pareto_todas_geracoes.png')
    plt.show()