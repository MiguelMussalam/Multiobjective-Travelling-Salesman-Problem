from AGFuncs import *
import numpy as np

if __name__ == '__main__':
    
    iniciaPopulacao(população)
    
    NUM_GERACOES = 50  # aumente para ver melhor a evolução

    for geracao in range(NUM_GERACOES):
        print(f"\n=== GERAÇÃO {geracao} ===")

        # CORREÇÃO: Calcular fitness da população ATUAL
        fitness_combinado = fitness(população,nota_pop)

        # Exibe população e fitness
        for i in range(TAM_POP):
            print(f"Cromossomo {i}: {população[i].astype(int)} | "
                f"Distância: {nota_pop[i,0]:.3f} | "
                f"Tempo: {nota_pop[i,1]:.3f} | "
                f"Pedágio: {nota_pop[i,2]:.3f}")

        # Encontra o melhor (menor distância)
        melhor_idx = np.argmin(fitness_combinado)
        print(f"Melhor da geração {geracao}: {população[melhor_idx].astype(int)} | "
            f"Fitness combinado: {fitness_combinado[melhor_idx]:.3f} | "
            f"Distância: {nota_pop[melhor_idx,0]:.3f} | "
            f"Tempo: {nota_pop[melhor_idx,1]:.3f} | "
            f"Pedágio: {nota_pop[melhor_idx,2]:.3f}")

        # CORREÇÃO: Criar nova geração DEPOIS de mostrar os resultados
        população = novaGeracao(população, fitness_combinado)
        
        # CORREÇÃO: Atualizar nota_pop para a nova população
        fitness_combinado = fitness(população,nota_pop)


