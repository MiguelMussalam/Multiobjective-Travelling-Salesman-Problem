from AGFuncs import *

if __name__ == '__main__':
    
    # Listas para gráficos (igual do professor)
    geracao_list = []
    melhor_list = []
    pior_list = []
    medio_list = []

    iniciaPopulacao(população)

    for geracao in range(NUM_GERACOES):
        print(f"\n=== GERAÇÃO {geracao} ===")

        # Calcula fitness da população ATUAL
        fitness_combinado = fitness(população,nota_pop)

        # Estatísticas para gráficos
        melhor_fitness = np.min(fitness_combinado)
        pior_fitness = np.max(fitness_combinado)
        medio_fitness = np.mean(fitness_combinado)
        
        # Exibe população e fitness
        for i in range(TAM_POP):
            print(f"Cromossomo {i}: {população[i].astype(int)} | "
                f"Distância: {nota_pop[i,0]:.3f} | "
                f"Tempo: {nota_pop[i,1]:.3f} | "
                f"Pedágio: {nota_pop[i,2]:.3f}")

        # Exibe o melhor fitness da geração
        melhor_idx = np.argmin(fitness_combinado)
        print(f"Melhor da geração {geracao}: {população[melhor_idx].astype(int)} | "
            f"Fitness combinado: {fitness_combinado[melhor_idx]:.3f} | "
            f"Distância: {nota_pop[melhor_idx,0]:.3f} | "
            f"Tempo: {nota_pop[melhor_idx,1]:.3f} | "
            f"Pedágio: {nota_pop[melhor_idx,2]:.3f}")

        geracao_list.append(geracao)
        melhor_list.append(melhor_fitness)
        pior_list.append(pior_fitness)
        medio_list.append(medio_fitness)

        # Cria nova geração 
        população = novaGeracao(população, fitness_combinado)
    
    # Gráfico de convergência
    plt.title("CONVERGENCIA AG")
    plt.plot(geracao_list, melhor_list, label="Melhor")
    plt.plot(geracao_list, pior_list, label="Pior")
    plt.plot(geracao_list, medio_list, label="Medio")
    plt.legend()
    plt.show()
        
    

